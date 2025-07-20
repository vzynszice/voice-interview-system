"""
pyttsx3 İstemcisi - Callback Bazlı Güvenilir Seslendirme (macOS Uyumlu)
"""
import asyncio
import threading
import queue
import time
import pyttsx3
from loguru import logger
from config import config

class Pyttsx3Client:
    def __init__(self):
        """
        Callback bazlı TTS istemcisi başlat
        """
        logger.info("pyttsx3 TTS istemcisi başlatılıyor (Callback Mod).")
        self.rate = config.models.tts_rate
        self.voice_id = None
        self.backend = "pyttsx3"
        
        # TTS işlemleri için queue
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.interruption_event = threading.Event()
        
        # Türkçe ses bul
        try:
            temp_engine = pyttsx3.init("nsss")
            voices = temp_engine.getProperty('voices')
            for voice in voices:
                if 'yelda' in voice.name.lower() or 'turkish' in voice.name.lower():
                    self.voice_id = voice.id
                    logger.success(f"Varsayılan Türkçe ses bulundu: {voice.name}")
                    break
            temp_engine.stop()
            del temp_engine
        except Exception as e:
            logger.warning(f"Türkçe ses ID'si bulunamadı: {e}")
        
        # TTS worker thread'i başlat
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        logger.info("TTS worker thread başlatıldı")

    def stop_playback(self):
        """
        Mevcut seslendirmeyi dışarıdan durdurur.
        """
        logger.info("Dışarıdan durdurma sinyali alındı.")
        self.interruption_event.set()
        # Kuyruktaki bekleyen diğer cümleleri de temizleyelim.
        with self.tts_queue.mutex:
            self.tts_queue.queue.clear()
            
    def _tts_worker(self):
        """
        Queue'dan gelen metinleri sırayla seslendir (callback ile)
        """
        logger.info("TTS worker başladı")
        
        # Tek bir engine oluştur
        try:
            engine = pyttsx3.init("nsss")
            engine.setProperty('rate', self.rate)
            if self.voice_id:
                engine.setProperty('voice', self.voice_id)
            
            # Callback'ler için state
            utterance_complete = threading.Event()
            
            def on_start(name):
                logger.debug(f"🎵 Seslendirme başladı")
                utterance_complete.clear()
            
            def on_end(name, completed):
                logger.debug(f"🎵 Seslendirme bitti (completed: {completed})")
                utterance_complete.set()
            
            # Callback'leri bağla
            engine.connect('started-utterance', on_start)
            engine.connect('finished-utterance', on_end)
            
        except Exception as e:
            logger.error(f"TTS engine başlatılamadı: {e}")
            return
        
        while not self.stop_event.is_set():
            try:
                # Queue'dan metin al
                task = self.tts_queue.get(timeout=1.0)
                
                if task is None:  # Stop sinyali
                    break
                
                text, complete_event = task
                start_time = time.time()
                logger.info(f"🔊 Seslendirme başlıyor: {text[:50]}...")
                
                # Metni seslendir
                utterance_complete.clear()
                engine.say(text)
                engine.startLoop(False)
                
                # Callback'in tetiklenmesini bekle (maksimum 30 saniye)
                timeout = 30.0
                waited = 0
                while not utterance_complete.is_set() and waited < timeout:
                    engine.iterate()
                    time.sleep(0.05)  # 50ms bekle
                    waited += 0.05
                
                engine.endLoop()
                
                duration = time.time() - start_time
                logger.info(f"✅ Seslendirme tamamlandı ({duration:.1f}s): {text[:50]}...")
                
                # Ses kartı buffer'ı için kısa bekleme
                time.sleep(0.2)
                
                # İşlemin tamamlandığını bildir
                complete_event.set()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker hatası: {e}")
                if 'complete_event' in locals():
                    complete_event.set()
        
        # Engine'i temizle
        try:
            engine.stop()
            del engine
        except:
            pass
        
        logger.info("TTS worker sonlandı")

    async def text_to_speech_and_play(self, text: str):
        """
        Metni seslendirme kuyruğuna ekle ve tamamlanmasını bekle
        """
        if not text or not text.strip():
            return
        
        # Tamamlanma event'i oluştur
        complete_event = threading.Event()
        
        # Queue'ya ekle
        logger.debug(f"📝 Queue'ya ekleniyor: {text[:50]}...")
        self.tts_queue.put((text, complete_event))
        
        # Event'in set edilmesini async olarak bekle
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, complete_event.wait, 60.0)  # 60 saniye timeout

    async def test_connection(self) -> bool:
        """TTS sisteminin çalışıp çalışmadığını test eder."""
        try:
            logger.info("pyttsx3 TTS testi yapılıyor...")
            await self.text_to_speech_and_play("Merhaba, bu bir test mesajıdır.")
            logger.info("✅ pyttsx3 TTS testi başarılı!")
            return True
        except Exception as e:
            logger.error(f"❌ pyttsx3 TTS test hatası: {e}")
            return False

    def cleanup(self):
        """TTS worker'ı durdur ve kaynakları temizle"""
        logger.info("TTS client temizleniyor...")
        self.stop_event.set()
        self.tts_queue.put(None)  # Stop sinyali
        if self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
        logger.info("TTS client temizlendi")

    def __del__(self):
        """Destructor - kaynakları temizle"""
        try:
            self.cleanup()
        except:
            pass