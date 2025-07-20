"""
Lokal Sesli Mülakat Sistemi - Ana Program

Tamamen lokal çalışan, gizlilik odaklı mülakat sistemi.
API anahtarlarına ihtiyaç duymaz, tüm işlemler lokalde yapılır.
"""

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

# Lokal konfigürasyon ve orchestrator
from config import config
from src.orchestrator.orchestrator import LocalInterviewOrchestrator
from src.orchestrator.schema import JobInfo, CandidateInfo
from src.orchestrator.state_manager import state_manager, InterviewState

# Lokal client'lar
from src.clients.ollama_client import OllamaClient
from src.clients.whisper_turbo_client import WhisperTurboClient
from src.clients.pyttsx3_client import Pyttsx3Client
from src.audio.audio_recorder import AudioRecorder

console = Console()


class LocalInterviewApp:
    """Lokal mülakat uygulaması ana sınıfı"""
    
    def __init__(self):
        self.console = console
        self.llm_client = OllamaClient()
        self.stt_client = WhisperTurboClient()
        self.tts_client = Pyttsx3Client()
        self.audio_recorder = AudioRecorder()
       


    async def run(self, args):
        """Ana uygulama döngüsü"""
        self._show_welcome()
        
        if args.cleanup_sessions:
            state_manager.cleanup_old_sessions()
            return
            
        try:
            job_info, candidate_info = None, None
            session_to_load = args.resume or await self._check_for_recoverable_sessions()
            
            if session_to_load:
                # Önceki oturumu yükle
                session = state_manager.load_session(session_to_load)
                if session:
                    job_info, candidate_info = session.job_info, session.candidate_info
                    self.console.print(f"\n✅ Oturum '{session_to_load}' yüklendi, devam ediliyor...")
                else:
                    self.console.print(f"[red]❌ Devam edilecek oturum bulunamadı: {session_to_load}[/red]")
                    return
            else:
                # Yeni mülakat başlat
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
        finally:
            self.audio_recorder.cleanup()
            self.tts_client.cleanup()  # Yeni eklenen satır
            logger.info("Uygulama sonlandırıldı, tüm ses kaynakları temizlendi.")
    
    def _show_welcome(self):
        """Hoş geldiniz mesajı"""
        panel = Panel(
            """🎙️  [bold cyan]Lokal Sesli Mülakat Sistemi[/bold cyan]
            
Tamamen bilgisayarınızda çalışan, gizlilik odaklı mülakat asistanı.
Tüm verileriniz lokalde kalır, hiçbir dış servise gönderilmez.
            
[dim]Modeller: Gemma3:1B (LLM) | Whisper (STT) | Lokal TTS[/dim]""",
            title="Hoş Geldiniz",
            border_style="cyan"
        )
        self.console.print(panel)
    
    async def _check_system(self) -> bool:
        """Lokal sistemleri kontrol et"""
        self.console.print("\n[bold]🔍 Sistem Kontrolleri[/bold]")
        
        all_ok = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Ollama kontrolü
            task = progress.add_task("[cyan]Ollama servisi kontrol ediliyor...", total=1)
            try:
                client = OllamaClient()
                if await client.test_connection():
                    progress.update(task, completed=1, description="[green]✅ Ollama hazır")
                else:
                    progress.update(task, completed=1, description="[red]❌ Ollama çalışmıyor")
                    all_ok = False
                await client.close()
            except Exception as e:
                progress.update(task, completed=1, description=f"[red]❌ Ollama hatası: {str(e)}")
                all_ok = False
            
            # Whisper kontrolü
            task = progress.add_task("[cyan]Whisper modeli kontrol ediliyor...", total=1)
            try:
                client = WhisperTurboClient()
                if await client.test_connection():
                    progress.update(task, completed=1, description="[green]✅ Whisper hazır")
                else:
                    progress.update(task, completed=1, description="[red]❌ Whisper yüklenemedi")
                    all_ok = False
            except Exception as e:
                progress.update(task, completed=1, description=f"[red]❌ Whisper hatası: {str(e)}")
                all_ok = False
            
            # TTS kontrolü
            task = progress.add_task("[cyan]TTS sistemi kontrol ediliyor...", total=1)
            try:
                client = Pyttsx3Client()
                if await client.test_connection():
                    progress.update(task, completed=1, description=f"[green]✅ TTS hazır ({client.backend})")
                else:
                    progress.update(task, completed=1, description="[red]❌ TTS çalışmıyor")
                    all_ok = False
            except Exception as e:
                progress.update(task, completed=1, description=f"[red]❌ TTS hatası: {str(e)}")
                all_ok = False
            
            # Ses sistemi kontrolü
            task = progress.add_task("[cyan]Ses sistemi kontrol ediliyor...", total=1)
            try:
                ### DEĞİŞİKLİK: Test için yeni bir nesne oluşturmak yerine mevcut olanı kullan ###
                if self.audio_recorder.input_device_index is not None:
                    progress.update(task, completed=1, description="[green]✅ Ses sistemi hazır")
                else:
                    raise Exception("Giriş cihazı bulunamadı.")
            except Exception as e:
                progress.update(task, completed=1, description=f"[red]❌ Ses hatası: {str(e)}")
                all_ok = False
        
        if not all_ok:
            self.console.print("\n[red]Bazı sistemler hazır değil. Lütfen şunları kontrol edin:[/red]")
            self.console.print("1. Ollama servisi çalışıyor mu? ([cyan]ollama serve[/cyan])")
            self.console.print("2. Gemma3:1b modeli yüklü mü? ([cyan]ollama pull gemma3:1b[/cyan])")
            self.console.print("3. PyAudio düzgün kurulu mu?")
            self.console.print("4. TTS sistemi (Coqui/pyttsx3) kurulu mu?")
        
        return all_ok
    
    async def _calibrate_audio(self):
        """Ses kalibrasyonu"""
        self.console.print("\n[bold]🎤 Ses Kalibrasyonu[/bold]")
        self.console.print("[yellow]3 saniye sessiz kalın, ortam gürültüsünü ölçüyoruz...[/yellow]")
        

        try:
            threshold = self.audio_recorder.calibrate_silence_threshold(duration=3.0)
            config.audio.silence_threshold = threshold
            self.console.print(f"✅ Kalibrasyon tamamlandı. Gürültü seviyesi: [green]{threshold}[/green]")
        except Exception as e:
            logger.error(f"Kalibrasyon sırasında bir hata oluştu: {e}")
            self.console.print("[red]❌ Kalibrasyon başarısız oldu. Varsayılan ayarlarla devam ediliyor.[/red]")
    
    async def _get_interview_data(self, args) -> tuple[JobInfo, CandidateInfo]:
        """Mülakat verilerini al"""
        self.console.print("\n[bold]📋 Mülakat Bilgileri[/bold]")
        
        if args.job_file:
            # Dosyadan yükle
            path = Path(args.job_file)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return JobInfo(**data['job']), CandidateInfo(**data['candidate'])
        
        if args.synthetic or Confirm.ask("Örnek mülakat verisi oluşturulsun mu?", default=True):
            return await self._generate_synthetic_data()
        
        return await self._manual_input()
    
    async def _generate_synthetic_data(self) -> tuple[JobInfo, CandidateInfo]:
        """Ollama ile sentetik veri üret"""
        positions = [
            "Python Developer",
            "Full Stack Developer",
            "Data Scientist",
            "DevOps Engineer",
            "Mobile Developer"
        ]
        
        position = Prompt.ask(
            "Pozisyon seçin",
            choices=positions,
            default="Python Developer"
        )
        
        # Basit template ile iş ilanı oluştur
        job_info = JobInfo(
            title=position,
            company="Tech Startup A.Ş.",
            requirements={
                'technical_skills': self._get_skills_for_position(position),
                'experience_years': 3,
                'education': 'Lisans'
            }
        )
        
        # Aday bilgileri
        candidate_name = Prompt.ask("Adınız", default="Demo Kullanıcı")
        
        candidate_info = CandidateInfo(
            name=candidate_name,
            current_position="Developer",
            years_experience=2,
            key_skills=["Python", "Git"]
        )
        
        return job_info, candidate_info
    
    def _get_skills_for_position(self, position: str) -> list[str]:
        """Pozisyona göre beceri listesi"""
        skills_map = {
            "Python Developer": ["Python", "Django", "PostgreSQL", "Docker", "REST API"],
            "Full Stack Developer": ["JavaScript", "React", "Node.js", "MongoDB", "HTML/CSS"],
            "Data Scientist": ["Python", "Pandas", "NumPy", "Scikit-learn", "SQL"],
            "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Linux"],
            "Mobile Developer": ["React Native", "Flutter", "Firebase", "REST API", "Git"]
        }
        return skills_map.get(position, ["Python", "Git", "SQL"])
    
    async def _manual_input(self) -> tuple[JobInfo, CandidateInfo]:
        """Manuel veri girişi"""
        # İş bilgileri
        job_info = JobInfo(
            title=Prompt.ask("Pozisyon başlığı"),
            company=Prompt.ask("Şirket adı"),
            requirements={
                'technical_skills': Prompt.ask("Gerekli beceriler (virgülle ayırın)").split(','),
                'experience_years': int(Prompt.ask("Minimum deneyim (yıl)", default="2")),
                'education': Prompt.ask("Eğitim seviyesi", default="Lisans")
            }
        )
        
        # Aday bilgileri
        candidate_info = CandidateInfo(
            name=Prompt.ask("Adınız"),
            current_position=Prompt.ask("Mevcut pozisyonunuz", default=""),
            years_experience=int(Prompt.ask("Toplam deneyim (yıl)", default="0")),
            key_skills=Prompt.ask("Ana becerileriniz (virgülle ayırın)").split(',')
        )
        
        return job_info, candidate_info
    
    async def _check_for_recoverable_sessions(self) -> str | None:
        """Yarıda kalmış mülakatları kontrol et"""
        recoverable = state_manager.get_recovery_info()
        if not recoverable:
            return None
        
        self.console.print("\n[bold yellow]📂 Yarıda Kalmış Mülakatlar[/bold yellow]")
        
        table = Table(title="Devam Edilebilir Oturumlar")
        table.add_column("No", style="cyan")
        table.add_column("Aday", style="white")
        table.add_column("Pozisyon", style="white")
        table.add_column("Oturum ID", style="dim")
        
        choices = {"0": "Yeni Mülakat"}
        for i, session in enumerate(recoverable, 1):
            table.add_row(
                str(i),
                session['candidate_name'],
                session['position'],
                session['session_id']
            )
            choices[str(i)] = session['session_id']
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "\nSeçiminiz (0: Yeni mülakat)",
            choices=list(choices.keys()),
            default="0"
        )
        
        return None if choice == "0" else choices[choice]
    
    async def _run_interview(self, job_info: JobInfo, candidate_info: CandidateInfo):
        """Mülakatı çalıştır"""
        self.console.print("\n[bold green]🎬 Mülakat Başlıyor![/bold green]")
        
        # State management
        session = state_manager.current_session
        if not session:
            session = state_manager.create_session(job_info, candidate_info)
        
        # Orchestrator oluştur
        orchestrator = LocalInterviewOrchestrator(
            job=job_info, 
            candidate=candidate_info,
            llm_client=self.llm_client,
            stt_client=self.stt_client,
            tts_client=self.tts_client,
            audio_recorder=self.audio_recorder
        )
        
        # Önceki state'i yükle
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
            self.console.print(
                f"\n[yellow]⏸️ Mülakat duraklatıldı. "
                f"Devam etmek için: python main.py --resume {session.session_id}[/yellow]"
            )
            raise
            
        except Exception as e:
            state_manager.mark_failed(str(e))
            raise
    
    def _show_results(self):
        """Mülakat sonuçlarını göster"""
        self.console.print("\n[bold]📊 Mülakat Sonuçları[/bold]")
        
        session = state_manager.current_session
        if not (session and session.state == InterviewState.COMPLETED):
            self.console.print("[yellow]Gösterilecek tamamlanmış mülakat yok.[/yellow]")
            return
        
        # Transkript dosyasını bul
        transcript_files = list(
            config.app.transcript_dir.glob(f"interview_{session.session_id.split('_')[1]}*.jsonl")
        )
        
        if transcript_files:
            latest = max(transcript_files, key=lambda p: p.stat().st_mtime)
            self.console.print(f"📄 Transkript: [green]{latest}[/green]")
            self.console.print(f"💬 Toplam soru-cevap: [cyan]{len(session.transcript) // 2}[/cyan]")
            self.console.print(f"⏱️  Toplam süre: [cyan]{session.total_duration/60:.1f} dakika[/cyan]")
        else:
            self.console.print("[yellow]Transkript dosyası bulunamadı.[/yellow]")
        
        self.console.print("\n[dim]Detaylı analiz için transkript dosyasını inceleyebilirsiniz.[/dim]")


def main():
    """Ana program giriş noktası"""
    parser = argparse.ArgumentParser(
        description="Lokal AI Destekli Sesli Mülakat Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py                    # Normal başlatma
  python main.py --synthetic        # Örnek veri ile başlat
  python main.py --job-file job.json  # Dosyadan yükle
  python main.py --resume session_123  # Önceki mülakatı devam ettir
  python main.py --cleanup-sessions    # Eski oturumları temizle
        """
    )
    
    parser.add_argument(
        '--job-file',
        type=str,
        help='Mülakat bilgilerini içeren JSON dosyası'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Sentetik veri üretimini kullan'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Sistem kontrollerini atla'
    )
    parser.add_argument(
        '--resume',
        type=str,
        metavar='SESSION_ID',
        help='Belirtilen oturumu devam ettir'
    )
    parser.add_argument(
        '--cleanup-sessions',
        action='store_true',
        help='7 günden eski oturum dosyalarını temizle'
    )
    
    args = parser.parse_args()
    
    # Uygulama başlat
    app = LocalInterviewApp()
    
    try:
        asyncio.run(app.run(args))
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Program kapatıldı.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Kritik hata: {str(e)}[/red]")
        logger.exception("Kritik hata")
        sys.exit(1)


if __name__ == "__main__":
    main()