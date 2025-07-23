import asyncio
import json
import sys
from pathlib import Path
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from loguru import logger

from config import config
from src.orchestrator.orchestrator import InterviewOrchestrator
from src.orchestrator.schema import JobInfo, CandidateInfo, QA
from src.orchestrator.state_manager import state_manager, InterviewState
from src.api_clients.groq_client import GroqClient
from src.api_clients.whisper_client import WhisperClient
from src.api_clients.elevenlabs_client import ElevenLabsClient
from src.api_clients.gct_client import GCTClient
from src.api_clients.chatgpt_client import ChatGPTClient
from src.audio.audio_recorder import AudioRecorder

console = Console()

class InterviewApp:
    def __init__(self):
        self.console = console
        self.synthetic_client = ChatGPTClient()
        
    async def run(self, args):
        self._show_welcome()
        
        if args.cleanup_sessions:
            state_manager.cleanup_old_sessions()
            return
            
        try:
            job_info, candidate_info = None, None
            session_to_load = args.resume or await self._check_for_recoverable_sessions()

            if session_to_load:
                session = state_manager.load_session(session_to_load)
                if session:
                    job_info, candidate_info = session.job_info, session.candidate_info
                    self.console.print(f"\n✅ Oturum '{session_to_load}' yüklendi, devam ediliyor...")
                else:
                    self.console.print(f"[red]❌ Devam edilecek oturum bulunamadı: {session_to_load}[/red]")
                    return
            else:
                if not args.skip_checks and not await self._check_system():
                    return
                
                await self._calibrate_audio()
                job_info, candidate_info = await self._get_interview_data(args)
            
            if job_info and candidate_info:
                await self._run_interview(job_info, candidate_info)
                self._show_results()
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  İşlem kullanıcı tarafından iptal edildi.[/yellow]")
        except Exception as e:
            logger.exception("Beklenmeyen hata")
            self.console.print(f"[red]❌ Hata: {str(e)}[/red]")

    async def _check_for_recoverable_sessions(self) -> str | None:
        recoverable = state_manager.get_recovery_info()
        if not recoverable: return None
        
        self.console.print("\n[bold yellow]📂 Kurtarılabilir Oturumlar Bulundu![/bold yellow]")
        table = Table(title="Yarıda Kalmış Mülakatlar")
        table.add_column("No", style="cyan")
        table.add_column("Aday", style="white")
        table.add_column("Oturum ID", style="dim")
        
        choices = {"0": "Yeni Mülakat Başlat"}
        for i, session in enumerate(recoverable, 1):
            table.add_row(str(i), session['candidate_name'], session['session_id'])
            choices[str(i)] = session['session_id']
        
        self.console.print(table)
        choice_num = Prompt.ask("\nDevam etmek istediğiniz mülakatın numarasını girin (Yeni için 0)", choices=list(choices.keys()), default="0")
        return None if choice_num == "0" else choices[choice_num]
    
    def _show_welcome(self):
        panel = Panel("...\n        🎙️  [bold cyan]Sesli Mülakat Sistemi[/bold cyan]\n...\n", title="Hoş Geldiniz", border_style="cyan")
        self.console.print(panel)
    
    async def _check_system(self) -> bool:
        self.console.print("\n[bold]🔍 Sistem Kontrolleri[/bold]")
        clients = {"Groq": GroqClient, "Whisper": WhisperClient, "ElevenLabs": ElevenLabsClient, "GCT": GCTClient, "ChatGPT": ChatGPTClient}
        all_ok = True
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
            for name, client_class in clients.items():
                task = progress.add_task(f"[cyan]{name}[/cyan] kontrol ediliyor...", total=1)
                try:
                    client = client_class()
                    success = await client.test_connection()
                    progress.update(task, completed=1)
                    if not success: all_ok = False
                except Exception:
                    progress.update(task, completed=1)
                    all_ok = False
        return all_ok
        
    async def _calibrate_audio(self):
        self.console.print("\n[bold]🎤 Ses Kalibrasyonu[/bold] ([yellow]3sn sessiz kalın[/yellow])")
        recorder = AudioRecorder()
        try:
            threshold = recorder.calibrate_silence_threshold(duration=3.0)
            config.audio.silence_threshold = threshold
            self.console.print(f"✅ Kalibrasyon tamamlandı. Eşik değeri: [green]{threshold}[/green]")
        finally:
            recorder.cleanup()

    async def _get_interview_data(self, args) -> tuple[JobInfo, CandidateInfo]:
        self.console.print("\n[bold]📋 Mülakat Bilgileri[/bold]")
        if args.job_file:
            path = Path(args.job_file)
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            return JobInfo(**data['job']), CandidateInfo(**data['candidate'])
        if args.synthetic or Confirm.ask("Sentetik veri üretilsin mi?", default=True):
            return await self._generate_synthetic_data()
        return await self._manual_input()

    async def _generate_synthetic_data(self) -> tuple[JobInfo, CandidateInfo]:
        position = Prompt.ask("Pozisyon seçin", choices=["Senior Backend Developer", "Data Scientist"], default="Data Scientist")
        package = await self.synthetic_client.create_complete_interview_package(position_title=position)
        job_data = package['jobPost']['basicInfo']
        job_info = JobInfo(title=job_data['title'], company=package['company']['name'], requirements={'technical_skills': [s['skillName'] for s in package['jobPost']['requirements']['skills']['technical']]})
        candidate_info = CandidateInfo(name=Prompt.ask("Adınız", default="Demo User"), key_skills=["Python"])
        return job_info, candidate_info

    async def _manual_input(self) -> tuple[JobInfo, CandidateInfo]:
        job_info = JobInfo(title=Prompt.ask("Pozisyon"), company=Prompt.ask("Şirket"), requirements={'technical_skills': Prompt.ask("Gerekli Beceriler").split(',')})
        candidate_info = CandidateInfo(name=Prompt.ask("Adınız"), key_skills=Prompt.ask("Ana Beceriler").split(','))
        return job_info, candidate_info

    async def _run_interview(self, job_info: JobInfo, candidate_info: CandidateInfo):
        self.console.print("\n[bold green]🎬 Mülakat Başlıyor![/bold green]")
        orchestrator = InterviewOrchestrator(job_info, candidate_info)
        session = state_manager.current_session
        if not session:
            session = state_manager.create_session(job_info, candidate_info)
        orchestrator.transcript = session.transcript
        orchestrator.phase = session.current_phase
        session.state = InterviewState.IN_PROGRESS
        state_manager.save_state()
        
        try:
            await orchestrator.run()
            state_manager.mark_completed()
            self.console.print("\n[bold green]✅ Mülakat başarıyla tamamlandı![/bold green]")
        except KeyboardInterrupt:
            session.state = InterviewState.PAUSED
            state_manager.save_state()
            self.console.print(f"\n[yellow]⏸️ Mülakat duraklatıldı. Devam etmek için: python main.py --resume {session.session_id}[/yellow]")
            raise
        except Exception as e:
            state_manager.mark_failed(str(e))
            raise

    def _show_results(self):
        self.console.print("\n[bold]📊 Mülakat Sonuçları[/bold]")
        session = state_manager.current_session
        if session and session.state == InterviewState.COMPLETED:
            # ... sonuçları göster
            pass

def main():
    parser = argparse.ArgumentParser(description="AI Destekli Sesli Mülakat Sistemi")
    parser.add_argument('--job-file', type=str, help='Mülakat bilgilerini içeren JSON dosyası')
    parser.add_argument('--synthetic', action='store_true', help='Sentetik veri üretimini kullan')
    parser.add_argument('--skip-checks', action='store_true', help='Sistem kontrollerini atla')
    parser.add_argument('--resume', type=str, metavar='ID', help='Belirtilen oturum ID\'sini devam ettir')
    parser.add_argument('--cleanup-sessions', action='store_true', help='Eski oturum dosyalarını temizle')
    args = parser.parse_args()
    
    app = InterviewApp()
    
    try:
        asyncio.run(app.run(args))
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Uygulama kapatıldı.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Kritik hata: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
