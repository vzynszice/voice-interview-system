"""
OpenAI Whisper API İstemcisi - Ses-Metin Dönüşümü

Bu modül, OpenAI'nin Whisper modelini kullanarak sesi metne çevirir.
Whisper, özellikle Türkçe için çok başarılı sonuçlar veriyor ve
arka plan gürültüsüne karşı dayanıklı.

Önemli Özellikler:
1. Otomatik dil algılama (Türkçe için optimize edilmiş)
2. Gürültü filtreleme
3. Noktalama işaretlerini otomatik ekleme
4. Farklı ses formatlarını destekleme
"""

import io
import os
import wave
import numpy as np
from typing import Optional, Union, BinaryIO
import asyncio
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import soundfile as sf

from config import config


class WhisperClient:
    """
    OpenAI Whisper API ile ses-metin dönüşümünü yöneten istemci.
    
    Bu sınıf, ses dosyalarını veya ses akışlarını alır ve bunları
    yüksek doğrulukla metne çevirir. Türkçe desteği mükemmel seviyededir.
    """
    
    def __init__(self):
        """Whisper istemcisini başlat"""
        self.api_key = config.api.openai_api_key
        
        # Senkron ve asenkron istemciler
        self.client = OpenAI(api_key=self.api_key,organization=os.getenv("OPENAI_ORG"))
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Whisper ayarları
        self.model = "whisper-1"  # Şu an tek model bu
        self.language = "tr"      # Türkçe
        self.response_format = "text"  # Alternatif: json, srt, vtt
        
        # İstatistikler
        self.total_transcriptions = 0
        self.total_audio_duration = 0.0
        
        logger.info(f"Whisper istemcisi başlatıldı. Dil: {self.language}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
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
            prompt: İsteğe bağlı yönlendirme metni (doğruluğu artırır)
            temperature: Çeşitlilik parametresi (0 = en deterministik)
            
        Returns:
            Transkript metni
            
        Raises:
            Exception: API hatası veya geçersiz ses formatı
        """
        try:
            logger.debug("Ses metne çevriliyor...")
            
            # Ses verisini hazırla
            audio_file = await self._prepare_audio_file(audio_data)
            if isinstance(audio_file[1], (bytes, bytearray)) and len(audio_file[1]) < 2000:
                raise ValueError("Kayıt çok kısa, transkripsiyon atlandı")
            # API çağrısı
            response = await self.async_client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
                response_format=self.response_format,
                prompt=prompt,
                temperature=temperature
            )
            
            # İstatistikleri güncelle
            self.total_transcriptions += 1
            
            # Metni temizle ve döndür
            transcript = self._clean_transcript(response)
            
            logger.info(f"Transkripsiyon başarılı. Uzunluk: {len(transcript)} karakter")
            return transcript
            
        except Exception as e:
            logger.error(f"Whisper API hatası: {e}")
            raise
    
    async def _prepare_audio_file(
        self,
        audio_data: Union[bytes, BinaryIO, Path]
    ) -> tuple:
        """
        Ses verisini API'nin kabul edeceği formata dönüştür.
        
        Whisper API, belirli formatlarda ses kabul eder.
        Bu metod, farklı kaynaklardan gelen sesi uygun formata çevirir.
        """
        if isinstance(audio_data, Path):
            # Dosya yolu verilmiş
            if not audio_data.exists():
                raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_data}")
            
            # Dosya formatını kontrol et
            supported_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
            if audio_data.suffix.lower() not in supported_formats:
                # Desteklenmeyen format, WAV'a çevir
                audio_data = await self._convert_to_wav(audio_data)
            
            with open(audio_data, 'rb') as f:
                return (audio_data.name, f.read(), 'audio/wav')
                
        elif isinstance(audio_data, bytes):
            # Bytes verisi, WAV formatında olduğunu varsayıyoruz
            # API için dosya benzeri nesne oluştur
            return ('audio.wav', audio_data, 'audio/wav')
            
        elif hasattr(audio_data, 'read'):
            # Dosya benzeri nesne
            content = audio_data.read()
            if hasattr(audio_data, 'name'):
                name = audio_data.name
            else:
                name = 'audio.wav'
            return (name, content, 'audio/wav')
            
        else:
            raise ValueError(f"Desteklenmeyen ses verisi tipi: {type(audio_data)}")
    
    async def _convert_to_wav(self, audio_path: Path) -> Path:
        """
        Ses dosyasını WAV formatına çevir.
        
        Whisper her formatı kabul etmez. Bu metod, desteklenmeyen
        formatları WAV'a çevirir.
        """
        logger.debug(f"Ses dosyası WAV'a çevriliyor: {audio_path}")
        
        # Çıktı dosyası
        wav_path = audio_path.with_suffix('.wav')
        
        # soundfile ile dönüştür
        data, sample_rate = sf.read(str(audio_path))
        sf.write(str(wav_path), data, sample_rate, subtype='PCM_16')
        
        return wav_path
    
    def _clean_transcript(self, response: Union[str, dict]) -> str:
        """
        Transkript metnini temizle ve düzelt.
        
        Whisper bazen gereksiz boşluklar veya karakterler ekleyebilir.
        Bu metod, metni temizler ve okunabilir hale getirir.
        """
        # Response string ise direkt kullan
        if isinstance(response, str):
            text = response
        else:
            # JSON response ise text alanını al
            text = response.get('text', '')
        
        # Temizleme işlemleri
        text = text.strip()
        
        # Çoklu boşlukları tek boşluğa çevir
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Noktalama düzeltmeleri (Türkçe için)
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        return text
    
    async def transcribe_stream(
        self,
        audio_stream: asyncio.Queue,
        chunk_duration: float = 5.0
    ) -> asyncio.Queue:
        """
        Ses akışını gerçek zamanlı olarak metne çevir.
        
        Bu metod, canlı ses akışını alır ve parça parça metne çevirir.
        Uzun mülakatlar için idealdir.
        
        Args:
            audio_stream: Ses chunk'larını içeren queue
            chunk_duration: Her chunk'ın süresi (saniye)
            
        Returns:
            Transkript chunk'larını içeren queue
        """
        transcript_queue = asyncio.Queue()
        
        async def process_chunks():
            buffer = bytearray()
            
            while True:
                try:
                    # Ses chunk'ını al
                    chunk = await asyncio.wait_for(
                        audio_stream.get(),
                        timeout=30.0  # 30 saniye timeout
                    )
                    
                    if chunk is None:  # Akış sonu sinyali
                        break
                    
                    buffer.extend(chunk)
                    
                    # Buffer yeterince dolduğunda işle
                    if len(buffer) >= config.audio.sample_rate * chunk_duration * 2:  # 16-bit audio
                        # WAV header ekle
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
        """
        Ham ses verisinden WAV dosyası oluştur.
        
        Mikrofondan gelen ham ses verisi WAV header'a ihtiyaç duyar.
        Bu metod, header'ı ekleyerek geçerli bir WAV dosyası oluşturur.
        """
        # WAV dosyası için BytesIO buffer
        buffer = io.BytesIO()
        
        # WAV writer oluştur
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(config.audio.channels)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
        
        # Buffer'ı başa sar
        buffer.seek(0)
        return buffer.read()
    
    def create_context_prompt(self, context: dict) -> str:
        """
        Transkripsiyon doğruluğunu artırmak için bağlam prompt'u oluştur.
        
        Whisper'a bağlam vermek, özellikle özel isimler ve terimler
        için doğruluğu artırır.
        
        Args:
            context: Mülakat bağlamı (pozisyon, şirket adı vs.)
            
        Returns:
            Bağlam prompt'u
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
        """API kullanım istatistiklerini döndür"""
        return {
            "total_transcriptions": self.total_transcriptions,
            "total_audio_duration": f"{self.total_audio_duration:.1f} saniye",
            "average_duration": (
                f"{self.total_audio_duration / self.total_transcriptions:.1f} saniye"
                if self.total_transcriptions > 0 else "N/A"
            )
        }
    
    async def test_connection(self) -> bool:
        """
        API bağlantısını test et.
        
        Küçük bir ses örneği göndererek API'nin çalıştığını doğrular.
        """
        try:
            logger.info("Whisper API bağlantısı test ediliyor...")
            
            # Test için küçük bir sessizlik WAV dosyası oluştur
            sample_rate = 16000
            duration = 0.5  # 0.5 saniye
            samples = int(sample_rate * duration)
            
            # Sessizlik (sıfırlar)
            silence = np.zeros(samples, dtype=np.int16)
            
            # WAV olarak kodla
            buffer = io.BytesIO()
            sf.write(buffer, silence, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            
            # API'ye gönder
            response = await self.transcribe_audio(
                buffer.read(),
                prompt="This is a test audio with silence."
            )
            
            logger.info("✅ Whisper API bağlantısı başarılı!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Whisper API bağlantı hatası: {e}")
            return False


# Test için
if __name__ == "__main__":
    async def test():
        client = WhisperClient()
        
        # Bağlantı testi
        if await client.test_connection():
            print("✅ Whisper bağlantısı başarılı!")
            
            # İstatistikler
            print(f"\nİstatistikler: {client.get_statistics()}")
        else:
            print("❌ Whisper bağlantısı başarısız!")
    
    asyncio.run(test())