"""
Lokal Mülakat Orchestrator'ı

Bu modül, tamamen lokal çalışan sesli mülakat sisteminin
ana koordinatörüdür. Tüm lokal modelleri kullanır.
"""

import asyncio
import time
import json
from datetime import datetime
import random
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import config

# Lokal client'ları import et
from src.clients.ollama_client import OllamaClient
from src.clients.whisper_turbo_client import WhisperTurboClient  
from src.clients.pyttsx3_client import Pyttsx3Client
from src.clients.argos_translator_client import ArgosTranslatorClient
from src.audio.audio_recorder import AudioRecorder
from src.orchestrator.schema import JobInfo, CandidateInfo, QA
from src.orchestrator.state_manager import state_manager
from src.orchestrator.text_parser import SentenceParser

class LocalInterviewOrchestrator:
    """Lokal modelleri kullanarak mülakat yöneten ana sınıf"""
    
    def __init__(
        self, 
        job: JobInfo, 
        candidate: CandidateInfo,
        llm_client: OllamaClient,
        stt_client: WhisperTurboClient,
        tts_client: Pyttsx3Client,
        audio_recorder: AudioRecorder
    ):
        """
        Orchestrator'ı başlatır ve gerekli tüm bileşenleri dışarıdan alır.
        
        Args:
            job: Mülakat yapılan iş bilgileri.
            candidate: Aday bilgileri.
            llm_client: LLM (Ollama) ile iletişim kuran istemci.
            stt_client: STT (Whisper) ile iletişim kuran istemci.
            tts_client: TTS (pyttsx3) ile iletişim kuran istemci.
            audio_recorder: Ses kaydı yapan istemci.
        """
        self.job_info = job
        self.candidate_info = candidate
        self.config = config
        self.console = Console()
   
        # State (Durum) yönetimi
        self.state_manager = state_manager
        self.session = self.state_manager.current_session
        if not self.session:
            raise ValueError("Orchestrator başlatılırken aktif bir oturum bulunamadı.")
        
        # Mülakat ilerleme bilgileri
        self.phase = self.session.current_phase
        self.transcript = self.session.transcript
        self.question_count = len([qa for qa in self.transcript if qa.role == 'ai'])
        
        # Diğer yardımcılar
        self.output_filename_base = f"interview_{self.session.session_id.split('_')[1]}"
        self.start_time = self.session.started_at or datetime.now()
        self.parser = SentenceParser()

        # --- Mimari Düzeltme: İstemcileri dışarıdan al, kendin oluşturma! ---
        # Bu sayede istemciler program boyunca sadece bir kez oluşturulur.
        self.llm_client = llm_client
        self.stt_client = stt_client
        self.tts_client = tts_client
        self.audio_recorder = audio_recorder
        
        # Opsiyonel çeviri istemcisi
        self.translator = None
        if config.interview.enable_translation:
            logger.info("Lokal çeviri sistemi (Argos Translate) başlatılıyor...")
            self.translator = ArgosTranslatorClient(from_code="en", to_code="tr")
            
        logger.success("Orchestrator tüm sistemlerle başarıyla başlatıldı.")
    
    
    
    async def _ask_ai_question(self) -> str:
        """LLM'den soru üretir, gerekirse çevirir."""
        logger.info(f"'{self.phase}' aşaması için soru üretiliyor...")
        
        try:
            # Prompt'u İngilizce olarak hazırla (config'den)
            messages = self.llm_client.create_interview_prompt(
                job_info=self.job_info.model_dump(),
                candidate_info=self.candidate_info.model_dump(),
                phase=self.phase,
                previous_qa=[qa.model_dump() for qa in self.transcript]
            )
            
            # LLM'den yanıt al
            question = await self.llm_client.generate_response(messages)
            
            # Gerekirse ARGOS ile çevir
            if self.translator:
                logger.info(f"Üretilen İngilizce soru: {question}")
                question = self.translator.translate(question)
            
            question = question.strip().replace("\"", "")
            if not question.endswith('?'): question += '?'
            
            logger.info(f"Nihai Türkçe soru: {question}")
            return question
            
        except Exception as e:
            logger.error(f"Soru üretme hatası: {e}")
            return self._get_fallback_question()
    
    
    def _get_fallback_question(self) -> str:
        """Yedek sorular (Türkçe)"""
        fallback_questions = {
            "warmup": [
                "Kısaca kendinizden ve bu pozisyona olan ilginizden bahseder misiniz?",
                "Kariyer yolculuğunuzdan biraz bahseder misiniz?"
            ],
            "technical": [
                "En son üzerinde çalıştığınız projeden ve kullandığınız teknolojilerden bahseder misiniz?",
                "Karşılaştığınız teknik bir zorluğu ve çözüm yaklaşımınızı anlatır mısınız?"
            ],
            "behavioral": [
                "Ekip çalışması yaptığınız bir projeden ve katkınızdan bahseder misiniz?",
                "Stresli bir durumu nasıl yönettiğinizi bir örnekle anlatır mısınız?"
            ],
            "situational": [
                "Bir projenin son teslim tarihine yetişmeyeceğini fark etseniz ne yapardınız?",
                "Ekip üyelerinizden biri ile fikir ayrılığı yaşasanız nasıl bir yaklaşım sergilerdiniz?"
            ],
            "closing": [
                "Bizimle ilgili merak ettiğiniz sorular var mı?",
                "Bu pozisyon veya şirketimiz hakkında sormak istediğiniz bir şey var mı?"
            ]
        }
        
        
        questions = fallback_questions.get(self.phase, fallback_questions["technical"])
        return random.choice(questions)


    async def _speak(self, text: str) -> bool:
        """
        Metni cümlelere ayırarak sırayla seslendirir.
        Bu metot, seslendirmenin kesintiye uğrayıp uğramadığını geri döndürür.
        DÖNÜŞ DEĞERİ: True (tüm cümleler başarıyla seslendirildi), False (kesintiye uğradı).
        """
        try:
            if not text or not text.strip():
                logger.warning("Seslendirilecek metin bulunamadı.")
                return True # Boş metin başarıyla tamamlanmış sayılır

            # Metni cümlelere ayır
            sentences = self.parser.parse_sentences(text)
            
            if not sentences:
                logger.warning("Metin cümlelere ayrılamadı.")
                return True
                
            logger.info(f"Metin {len(sentences)} cümleye ayrıldı")
            
            # Her cümleyi sırayla seslendir
            for i, sentence in enumerate(sentences):
                if sentence.strip():  # Boş cümleleri atla
                    # İlk cümle için özel mesaj göster
                    if i == 0:
                        self.console.print(Text("🔊 Yapay zeka konuşuyor...", style="dim cyan"))
                    
                    logger.debug(f"Seslendiriliyor ({i+1}/{len(sentences)}): {sentence}")
                    
                    # Cümleyi seslendir ve sonucunu kontrol et
                    completed_successfully = await self.tts_client.text_to_speech_and_play(sentence)
                    
                    # Eğer seslendirme kesintiye uğradıysa, döngüden hemen çık
                    if not completed_successfully:
                        logger.warning("Konuşma kesintiye uğradığı için _speak metodu sonlandırılıyor.")
                        return False # Kesintiye uğradığını bildir

                    # Cümleler arası kısa bir duraklama (son cümle değilse)
                    if i < len(sentences) - 1:
                        await asyncio.sleep(0.2)
            
            logger.info("Tüm cümleler başarıyla seslendirildi.")
            return True # Başarıyla tamamlandığını bildir
            
        except Exception as e:
            logger.error(f"Seslendirme hatası: {e}")
            return False # Hata durumunda da kesilmiş gibi davran
    
    async def _listen(self) -> str:
        """
        Kullanıcıyı dinler, sesi kaydeder ve metne çevirir.
        Bu metod, sadece ana işlevine odaklanmıştır.
        """
        try:
            # 1. Kullanıcıdan ses kaydını al.
            #    AudioRecorder bu noktada "Lütfen konuşun..." gibi bir mesaj gösterebilir
            #    ve sessizlik algıladığında kaydı otomatik olarak bitirir.
            wav_data = await self.audio_recorder.start_recording()
            
            # 2. Kaydın geçerli olup olmadığını kontrol et.
            #    Eğer kullanıcı hiç konuşmadıysa veya kayıt çok kısaysa, boş bir metin döndür.
            if not wav_data or len(wav_data) < 2000: # 2KB'dan küçükse (yaklaşık 0.1 saniye) geçersiz say
                logger.info("Geçerli bir ses kaydı alınamadı, boş cevap olarak kabul ediliyor.")
                return ""
                
            # 3. Ses verisini Whisper istemcisine göndererek metne çevir.
            self.console.print("[yellow]Cevabınız metne dönüştürülüyor...[/yellow]")
            transcript = await self.stt_client.transcribe_audio(wav_data)
            
            return transcript
                
        except Exception as e:
            logger.error(f"Dinleme veya metne çevirme hatası: {e}")
            self.console.print(f"[red]⚠️ Sesiniz algılanamadı veya işlenemedi.[/red]")
            return "[Ses algılanamadı]"
    
    # src/orchestrator/orchestrator.py

    async def run(self):
        """
        Ana mülakat döngüsünü yönetir. 
        Yarıda kalmış oturumları destekler ve her adımdan sonra durumu kaydeder.
        Bu versiyon, kullanıcının AI konuşurken araya girmesini (interruptibility) destekler.
        """
        try:
            # Mülakatın başı mı yoksa devamı mı olduğunu kontrol et
            if not self.transcript:
                welcome_msg = f"Merhaba {self.candidate_info.name}, {self.job_info.company} şirketinin {self.job_info.title} pozisyonu için bugün sizinle görüşeceğim. Hazırsanız başlayalım."
                self.console.print(Panel(Text(welcome_msg, style="bold green"), title="🎙️ Mülakat Başlıyor", border_style="green"))
                await self._speak(welcome_msg)
            else:
                self.console.print(Panel(f"✅ Mülakata '{self.phase}' aşamasından devam ediliyor...", title="🎙️ Mülakat Devam Ediyor", border_style="yellow"))
            
            # Her mülakat aşaması için döngüye gir
            for phase in self.config.interview.interview_phases:
                # Tamamlanmış aşamaları atlama mantığı
                if phase in self.session.completed_phases:
                    logger.info(f"'{phase}' aşaması zaten tamamlanmış, atlanıyor.")
                    continue
                
                # Mevcut durumu güncelle
                self.phase = phase
                self.session.current_phase = phase
                num_questions = self.config.interview.questions_per_phase.get(phase, 1)
                self.console.print(Panel(f"[bold cyan]Aşama: {phase.capitalize()} ({num_questions} Soru)[/bold cyan]"))

                # Bu aşamadaki her soru için döngüye gir
                for i in range(num_questions):
                    self.question_count += 1
                    
                    # 1. Soru üret ve ekrana yazdır
                    question = await self._ask_ai_question()
                    self.console.print(Text(f"\n🤖 AI: {question}", style="bright_blue"))

                    # 2. Paralel görevleri oluştur: Biri konuşur, diğeri kesinti için dinler
                    speak_task = asyncio.create_task(self._speak(question))
                    interrupt_task = asyncio.create_task(self.audio_recorder.listen_for_interruption())

                    # 3. Hangi görevin önce biteceğini bekle (AI konuşmayı bitirecek mi, kullanıcı mı araya girecek?)
                    done, pending = await asyncio.wait(
                        {speak_task, interrupt_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # 4. Sonucu işle
                    if interrupt_task in done:
                        # Kullanıcı araya girdi!
                        self.console.print(Text("🎤 Araya girdiniz...", style="yellow"))
                        # Hemen TTS'i sustur ve bekleyen cümleleri temizle
                        self.tts_client.stop_playback()
                        # Hala çalışan diğer görevi (speak_task) iptal et
                        for task in pending:
                            task.cancel()
                    else: # speak_task in done
                        # AI konuşmasını bitirdi, artık kesinti için dinlemeye gerek yok.
                        # Diğer görevi (interrupt_task) iptal et.
                        for task in pending:
                            task.cancel()
                    
                    # 5. Şimdi kullanıcının tam cevabını dinle
                    self.console.print(Text("💬 Lütfen yanıtlayın...", style="dim"))
                    answer = await self._listen() # Bu, orijinal tam kayıt metodu
                    self.console.print(Text(f"👤 Siz: {answer or '[Sessizlik]'}", style="bright_green"))
                    
                    # 6. Her adımdan sonra durumu kaydet
                    ts = time.time()
                    self.transcript.append(QA(role="ai", text=question, ts=ts))
                    self.transcript.append(QA(role="human", text=answer, ts=ts))
                    self.session.transcript = self.transcript
                    self.session.current_question_index = i + 1
                    self.state_manager.save_state()
                    logger.success(f"Durum kaydedildi. Soru-Cevap: {len(self.transcript)}")
                
                # Faz tamamlandı olarak işaretle ve durumu tekrar kaydet
                self.session.completed_phases.append(phase)
                self.state_manager.save_state()
            
            # Mülakat bitti mesajı
            closing_msg = "Mülakat tamamlandı. Zaman ayırdığınız için teşekkür ederiz. En kısa sürede size dönüş yapacağız."
            self.console.print(Panel(Text(closing_msg, style="bold green"), title="✅ Mülakat Tamamlandı", border_style="green"))
            await self._speak(closing_msg)
            
            # Oturumun durumunu "tamamlandı" olarak güncelle
            self.state_manager.mark_completed()
            
        finally:
            # Ne olursa olsun (hata veya başarı), bu blok çalışır
            logger.info("Mülakat döngüsü tamamlandı, temizlik yapılıyor...")
            self._show_statistics()
            await self._save_transcript()
            await self._cleanup()
    
    async def _save_transcript(self):
        """Mülakat transkriptini kaydet"""
        try:
            interview_duration = time.time() - self.start_time.timestamp()
            
            transcript_path = self.config.app.transcript_dir / f"{self.output_filename_base}.jsonl"
            
            with transcript_path.open("w", encoding="utf-8") as f:
                # Metadata
                metadata = {
                    "interview_id": self.output_filename_base,
                    "date": self.start_time.isoformat(),
                    "duration_seconds": interview_duration,
                    "total_questions": self.question_count,
                    "job_info": self.job_info.model_dump(),
                    "candidate_info": self.candidate_info.model_dump(),
                    "system_info": {
                        "llm": config.models.ollama_model,
                        "stt": f"whisper-{config.models.whisper_model}",
                        "tts": config.models.tts_backend,
                        "streaming_tts": config.app.enable_streaming_tts
                    }
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                
                # Transkript
                for qa in self.transcript:
                    f.write(qa.model_dump_json() + "\n")
            
            logger.info(f"📄 Transkript kaydedildi: {transcript_path}")
            
        except Exception as e:
            logger.error(f"Transkript kaydetme hatası: {e}")
    
    def _show_statistics(self):
        """Mülakat istatistiklerini göster"""
        duration = time.time() - self.start_time.timestamp()
        
        stats = {
            "Toplam Süre": f"{duration/60:.1f} dakika",
            "Soru Sayısı": self.question_count,
            "Ortalama Soru Süresi": f"{duration/self.question_count:.1f} saniye" if self.question_count > 0 else "N/A",
            "LLM Model": config.models.ollama_model,
            "STT Model": f"Whisper {config.models.whisper_model}",
            "TTS Backend": self.tts_client.backend
        }
        
        self.console.print("\n[bold]📊 Mülakat İstatistikleri[/bold]")
        for key, value in stats.items():
            self.console.print(f"  {key}: [cyan]{value}[/cyan]")
    
    async def _cleanup(self):
        """Kaynakları temizle"""
        try:
            # Audio bileşenleri
            self.audio_recorder.cleanup()
            
            # Async client'ları kapat
            if hasattr(self.llm_client, 'close'):
                await self.llm_client.close()
                
        except Exception as e:
            logger.error(f"Temizlik hatası: {e}")


if __name__ == "__main__":
    async def test():
        # Gerekli importları fonksiyon içine alalım
        from src.orchestrator.state_manager import state_manager
        from src.orchestrator.schema import JobInfo, CandidateInfo
        
        # Test verileri aynı kalıyor
        job = JobInfo(
            title="Python Developer",
            company="Tech Startup",
            requirements={"technical_skills": ["Python", "FastAPI", "PostgreSQL"]}
        )
        candidate = CandidateInfo(
            name="Test Kullanıcı",
            current_position="Jr. Developer",
            years_experience=2,
            key_skills=["Python", "Django"]
        )
        
        # --- KRİTİK VE YENİ ADIMLAR ---
        # 1. Önce StateManager ile bir oturum oluştur.
        # Bu, orchestrator'ın ihtiyaç duyduğu 'current_session'ı ayarlar.
        print("📝 StateManager ile yeni bir test oturumu oluşturuluyor...")
        state_manager.create_session(job_info=job, candidate_info=candidate)
        print(f"✅ Oturum oluşturuldu: {state_manager.current_session.session_id}")
        # --- KRİTİK ADIMLARIN SONU ---
        
        # 2. Artık Orchestrator'ı güvenle başlatabiliriz.
        # __init__ metodu artık çalışacak çünkü aktif bir oturum var.
        orchestrator = LocalInterviewOrchestrator(job, candidate)
        
        # 3. Mülakatı çalıştır
        await orchestrator.run()
    
    # Kodu çalıştır
    asyncio.run(test())