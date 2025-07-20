"""
Whisper Turbo (faster-whisper) İstemcisi - Lokal Ses-Metin Dönüşümü

Bu modül, OpenAI Whisper API'nin yerini alacak şekilde,
faster-whisper kullanarak lokal ve hızlı transkripsiyon sağlar.
"""

import io
import wave
import numpy as np
from typing import Optional, Union, BinaryIO
import asyncio
from pathlib import Path
from faster_whisper import WhisperModel
import soundfile as sf
from loguru import logger

from src.utils.error_handlers import api_retry_handler
from config import config


class WhisperTurboClient:
    """
    Faster-whisper ile lokal ses-metin dönüşümünü yöneten istemci.
    
    OpenAI Whisper API'ye göre çok daha hızlı ve ücretsiz.
    """
    
    def __init__(self):
        """Whisper Turbo istemcisini başlat"""
        # Model seçimi - Türkçe için base yeterli ve hızlı
        self.model_path_or_size = config.models.whisper_model
        self.device = config.models.whisper_device
        self.compute_type = config.models.whisper_compute_type
        self.language = config.models.whisper_language
        self.task = "transcribe"
        # Modeli yükle
        logger.info(f"Whisper modeli yükleniyor: {self.model_path_or_size}")
        self.model = WhisperModel(
            self.model_path_or_size, # DEĞİŞTİ
            device=self.device,
            compute_type=self.compute_type,
            # Model indirme yolunu da config'den alıyoruz
            download_root=str(config.models.models_dir / "whisper")
        )
        
        # İstatistikler
        self.total_transcriptions = 0
        self.total_audio_duration = 0.0
        
        logger.info(f"Whisper Turbo istemcisi başlatıldı. Model: {self.model_path_or_size}")
    
    @api_retry_handler()
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, BinaryIO, Path],
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:
        """
        Ses verisini metne çevir.
        
        Args:
            audio_data: Ses verisi (bytes, dosya handle veya dosya yolu)
            prompt: İsteğe bağlı yönlendirme metni
            temperature: Çeşitlilik parametresi (0 = en deterministik)
            
        Returns:
            Transkript metni
        """
        try:
            logger.debug("Ses metne çevriliyor...")
            
            # Ses verisini hazırla
            audio_path = await self._prepare_audio_file(audio_data)
            
            # faster-whisper ile transkripsiyon (CPU-bound işlem)
            def transcribe():
                segments, info = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task=self.task,
                    initial_prompt=prompt,
                    temperature=temperature,
                    vad_filter=True,  # VAD filtresi - sessizlikleri atla
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )
                
                # Segmentleri birleştir
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                return " ".join(text_parts), info
            
            # Async executor'da çalıştır
            loop = asyncio.get_event_loop()
            transcript, info = await loop.run_in_executor(None, transcribe)
            
            # İstatistikleri güncelle
            self.total_transcriptions += 1
            self.total_audio_duration += info.duration
            
            # Metni temizle
            transcript = self._clean_transcript(transcript)
            
            logger.info(f"Transkripsiyon başarılı. Süre: {info.duration:.1f}s, Metin: {len(transcript)} karakter")
            
            # Geçici dosyayı temizle
            if isinstance(audio_path, Path) and audio_path.name.startswith("temp_"):
                audio_path.unlink(missing_ok=True)
            
            return transcript
            
        except Exception as e:
            logger.error(f"Whisper Turbo hatası: {e}")
            raise
    
    async def _prepare_audio_file(
        self,
        audio_data: Union[bytes, BinaryIO, Path]
    ) -> Path:
        """
        Ses verisini faster-whisper'ın kabul edeceği formata dönüştür.
        
        faster-whisper dosya yolu bekler, bytes'ı geçici dosyaya yazarız.
        """
        if isinstance(audio_data, Path):
            return audio_data
            
        elif isinstance(audio_data, bytes):
            # Geçici dosya oluştur
            import uuid
            
            temp_dir = config.app.temp_dir
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"temp_{uuid.uuid4().hex[:8]}.wav"
            
            # WAV dosyası olarak kaydet
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            return temp_path
            
        elif hasattr(audio_data, 'read'):
            # Dosya benzeri nesne
            content = audio_data.read()
            return await self._prepare_audio_file(content)
            
        else:
            raise ValueError(f"Desteklenmeyen ses verisi tipi: {type(audio_data)}")
    
    def _clean_transcript(self, text: str) -> str:
        """
        Transkript metnini temizle ve düzelt.
        """
        if not text:
            return ""
        
        # Başta ve sonda boşlukları temizle
        text = text.strip()
        
        # Çoklu boşlukları tek boşluğa çevir
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Noktalama düzeltmeleri (Türkçe için)
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        # Whisper bazen tekrar eden kelimeler üretebilir
        # Basit tekrar temizleme
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word != prev_word:  # Aynı kelime art arda gelmiyorsa
                cleaned_words.append(word)
            prev_word = word
        
        return ' '.join(cleaned_words)
    
    async def transcribe_stream(
        self,
        audio_stream: asyncio.Queue,
        chunk_duration: float = 5.0
    ) -> asyncio.Queue:
        """
        Ses akışını gerçek zamanlı olarak metne çevir.
        
        Bu metod, canlı ses akışını alır ve parça parça metne çevirir.
        """
        transcript_queue = asyncio.Queue()
        
        async def process_chunks():
            buffer = bytearray()
            
            while True:
                try:
                    # Ses chunk'ını al
                    chunk = await asyncio.wait_for(audio_stream.get(), timeout=30.0)
                    
                    if chunk is None:  # Akış sonu sinyali
                        break
                    
                    buffer.extend(chunk)
                    
                    # Buffer yeterince dolduğunda işle
                    if len(buffer) >= config.audio.sample_rate * chunk_duration * 2:  # 16-bit audio
                        # WAV verisi oluştur
                        wav_data = self._create_wav_from_raw(
                            bytes(buffer),
                            config.audio.sample_rate
                        )
                        
                        # Transkribe et
                        transcript = await self.transcribe_audio(wav_data)
                        
                        # Kuyruğa ekle
                        await transcript_queue.put(transcript)
                        
                        # Buffer'ı temizle
                        buffer.clear()
                        
                except asyncio.TimeoutError:
                    logger.warning("Ses akışı zaman aşımı")
                    break
                except Exception as e:
                    logger.error(f"Chunk işleme hatası: {e}")
                    await transcript_queue.put(f"[HATA: {str(e)}]")
            
            # Son kalan veriyi işle
            if buffer:
                wav_data = self._create_wav_from_raw(
                    bytes(buffer),
                    config.audio.sample_rate
                )
                transcript = await self.transcribe_audio(wav_data)
                await transcript_queue.put(transcript)
            
            # Bitiş sinyali
            await transcript_queue.put(None)
        
        # Arka planda işlemeyi başlat
        asyncio.create_task(process_chunks())
        
        return transcript_queue
    
    def _create_wav_from_raw(self, raw_audio: bytes, sample_rate: int) -> bytes:
        """Ham ses verisinden WAV dosyası oluştur"""
        buffer = io.BytesIO()
        
        # WAV writer oluştur
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(config.audio.channels)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
        
        buffer.seek(0)
        return buffer.read()
    
    def create_context_prompt(self, context: dict) -> str:
        """
        Transkripsiyon doğruluğunu artırmak için bağlam prompt'u oluştur.
        """
        prompts = []
        
        if 'company' in context:
            prompts.append(f"Bu bir {context['company']} şirketi için iş mülakatı.")
        
        if 'position' in context:
            prompts.append(f"Pozisyon: {context['position']}.")
        
        if 'technical_terms' in context:
            prompts.append(f"Teknik terimler: {', '.join(context['technical_terms'])}")
        
        return ' '.join(prompts)
    
    def get_statistics(self) -> dict:
        """Kullanım istatistiklerini döndür"""
        return {
            "total_transcriptions": self.total_transcriptions,
            "total_audio_duration": f"{self.total_audio_duration:.1f} saniye",
            "average_duration": (
                f"{self.total_audio_duration / self.total_transcriptions:.1f} saniye"
                if self.total_transcriptions > 0 else "N/A"
            ),
            "model": self.model_path_or_size,
            "device": self.device
        }
    
    async def test_connection(self) -> bool:
        """
        Whisper modelinin yüklendiğini test et.
        """
        try:
            logger.info("Whisper Turbo testi yapılıyor...")
            
            # Test için küçük bir sessizlik oluştur
            sample_rate = 16000
            duration = 0.5
            samples = int(sample_rate * duration)
            
            # Sessizlik (düşük seviye gürültü ekle, tamamen sıfır olmasın)
            noise = np.random.normal(0, 0.001, samples).astype(np.float32)
            
            # WAV olarak kodla
            buffer = io.BytesIO()
            sf.write(buffer, noise, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            
            # Transkribe et
            response = await self.transcribe_audio(
                buffer.read(),
                prompt="Test audio"
            )
            
            logger.info("✅ Whisper Turbo testi başarılı!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Whisper Turbo test hatası: {e}")
            return False


# Test için
if __name__ == "__main__":
    async def test():
        client = WhisperTurboClient()
        
        # Model testi
        if await client.test_connection():
            print("✅ Whisper Turbo başarıyla yüklendi!")
            
            # İstatistikler
            print(f"\nİstatistikler: {client.get_statistics()}")
            
            # Gerçek bir ses dosyası varsa test et
            test_audio_path = Path("test_audio.wav")
            if test_audio_path.exists():
                print("\nGerçek ses dosyası transkribe ediliyor...")
                transcript = await client.transcribe_audio(test_audio_path)
                print(f"Transkript: {transcript}")
        else:
            print("❌ Whisper Turbo yüklenemedi!")
    
    asyncio.run(test())