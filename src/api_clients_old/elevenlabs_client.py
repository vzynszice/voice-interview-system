"""
ElevenLabs API İstemcisi - Metin-Ses Dönüşümü

Bu modül, ElevenLabs'in gelişmiş metin-ses dönüşüm teknolojisini kullanır.
ElevenLabs, özellikle doğal konuşma tonlaması ve duygu aktarımında
çok başarılıdır. Türkçe desteği de oldukça iyidir.

Özellikler:
1. Çoklu ses seçenekleri (kadın/erkek, genç/yaşlı)
2. Duygu ve ton kontrolü
3. Streaming ses üretimi
4. Ses kalitesi optimizasyonu
"""

import asyncio
from typing import Union, AsyncGenerator
from pathlib import Path
from src.audio import AudioPlayer
import aiohttp
from loguru import logger
from elevenlabs import AsyncElevenLabs, VoiceSettings, Voice
import numpy as np
import soundfile as sf

from config import config


class ElevenLabsClient:
    """
    ElevenLabs API ile metin-ses dönüşümünü yöneten istemci.
    
    Bu sınıf, metinleri yüksek kalitede, doğal ses dosyalarına çevirir.
    Mülakat sırasında AI'ın sorularını seslendirmek için kullanılır.
    """
    
    def __init__(self):
        """ElevenLabs istemcisini başlat"""
        self.api_key = config.api.elevenlabs_api_key
        self.voice_id = config.api.elevenlabs_voice_id
        
        # Async istemci
        self.client = AsyncElevenLabs(api_key=self.api_key)
        
        # Ses ayarları
        self.voice_settings = VoiceSettings(
            stability=0.75,        # Ses tutarlılığı (0-1)
            similarity_boost=0.85, # Ses benzerliği (0-1)
            style=0.5,            # Konuşma stili (0-1)
            use_speaker_boost=True # Ses kalitesi artırma
        )
        
        # Model seçimi
        self.model_id = "eleven_multilingual_v2"  # Türkçe desteği olan model
        
        # İstatistikler
        self.total_characters = 0
        self.total_requests = 0
        
        logger.info(f"ElevenLabs istemcisi başlatıldı. Ses ID: {self.voice_id}")
    
    async def text_to_speech(
        self,
        text: str,
        output_format = "wav",
        stream: bool = False
    ) -> Union[bytes, AsyncGenerator[bytes, None]]:
        """
        Metni sese çevir.
        
        Args:
            text: Sese çevrilecek metin
            output_format: Ses formatı (mp3_44100_128, pcm_16000, vb.)
            stream: Streaming kullanılsın mı?
            
        Returns:
            Ses verisi (bytes) veya stream
            
        Raises:
            Exception: API hatası durumunda
        """
        try:
            logger.debug(f"Metin sese çevriliyor. Uzunluk: {len(text)} karakter")
            
            # Metni temizle
            text = self._clean_text(text)
            
            # API çağrısı
            if stream:
                # Streaming response
                audio_stream = await self.client.generate(
                    text=text,
                    voice=Voice(
                        voice_id=self.voice_id,
                        settings=self.voice_settings
                    ),
                    model=self.model_id,
                    stream=True,
                    output_format=output_format
                )
                
                return self._handle_stream(audio_stream)
            else:
                # Normal response
                audio = await self.client.generate(
                    text=text,
                    voice=Voice(
                        voice_id=self.voice_id,
                        settings=self.voice_settings
                    ),
                    model=self.model_id,
                    stream=False,
                    output_format="pcm_16000"
                )
                if isinstance(audio, (bytes, bytearray)):
                    audio_bytes = audio
                else:                             # async_generator
                    audio_bytes = b"".join([chunk async for chunk in audio])
                # İstatistikleri güncelle
                self.total_requests += 1
                self.total_characters += len(text)
                
                  
                return audio_bytes
                
        except Exception as e:
            logger.error(f"ElevenLabs API hatası: {str(e)}")
            raise
    
    async def _handle_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming ses verisini işle.
        
        Streaming, uzun metinlerde ilk ses parçasının hızlıca
        oynatılmasını sağlar. Kullanıcı deneyimini iyileştirir.
        """
        try:
            async for chunk in audio_stream:
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming hatası: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Metni ses sentezi için temizle.
        
        ElevenLabs bazı karakterleri yanlış telaffuz edebilir.
        Bu metod, metni optimize eder.
        """
        # Çoklu boşlukları temizle
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Özel karakterleri düzelt
        replacements = {
            '...': '.',     # Üç nokta yerine tek nokta
            '!!': '!',      # Çoklu ünlem yerine tek
            '??': '?',      # Çoklu soru işareti yerine tek
            '\n': ' ',      # Satır sonlarını boşluğa çevir
            '\t': ' ',      # Tab'ları boşluğa çevir
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Başta ve sonda boşlukları temizle
        text = text.strip()
        
        # Çok uzun metinleri böl (API limiti)
        max_length = 5000  # ElevenLabs limiti
        if len(text) > max_length:
            logger.warning(f"Metin çok uzun ({len(text)} karakter), kısaltılıyor...")
            text = text[:max_length-3] + "..."
        
        return text
    
    async def text_to_speech_file(
        self,
        text: str,
        output_path: Union[str, Path],
        output_format: str = "mp3_44100_128"
    ) -> Path:
        """
        Metni ses dosyasına çevir ve kaydet.
        
        Args:
            text: Sese çevrilecek metin
            output_path: Çıktı dosya yolu
            output_format: Ses formatı
            
        Returns:
            Kaydedilen dosyanın yolu
        """
        output_path = Path(output_path)
        
        # Dizini oluştur
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sesi üret
        audio_data = await self.text_to_speech(text, output_format, stream=False)
        
        # Dosyaya yaz
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Ses dosyası kaydedildi: {output_path}")
        return output_path
    
    async def convert_format(
        self,
        audio_data: bytes,
        from_format: str,
        to_format: str
    ) -> bytes:
        """
        Ses formatını dönüştür.
        
        ElevenLabs MP3 döndürür ama bazen WAV gerekebilir.
        Bu metod format dönüşümü yapar.
        """
        if from_format == to_format:
            return audio_data
        
        # MP3'ten WAV'a dönüşüm örneği
        if from_format == "mp3" and to_format == "wav":
            # Geçici dosya kullan (soundfile MP3 okuyamaz)
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tmp_mp3.write(audio_data)
                tmp_mp3.flush()
                
                # ffmpeg veya benzeri araç gerekir
                # Şimdilik basit tutuyoruz
                logger.warning("Format dönüşümü için ffmpeg gerekli")
                return audio_data
        
        return audio_data
    
    async def get_voices(self) -> list:
        """
        Kullanılabilir sesleri listele.
        
        ElevenLabs'te farklı ses seçenekleri vardır.
        Bu metod mevcut sesleri listeler.
        """
        try:
            voices = await self.client.voices.get_all()
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                    "labels": voice.labels,
                    "preview_url": voice.preview_url
                }
                voice_list.append(voice_info)
            
            logger.info(f"{len(voice_list)} ses bulundu")
            return voice_list
            
        except Exception as e:
            logger.error(f"Ses listesi alınamadı: {e}")
            return []
    
    async def optimize_settings_for_interview(self):
        """
        Mülakat için ses ayarlarını optimize et.
        
        Mülakat ortamında net, anlaşılır ve profesyonel
        bir ses tonu önemlidir.
        """
        self.voice_settings = VoiceSettings(
            stability=0.85,         # Daha tutarlı ses
            similarity_boost=0.75,  # Daha doğal ton
            style=0.3,             # Daha nötr stil
            use_speaker_boost=True
        )
        
        logger.info("Ses ayarları mülakat için optimize edildi")
    
    def calculate_cost(self, text: str) -> float:
        """
        Tahmini maliyeti hesapla.
        
        ElevenLabs karakter başına ücret alır.
        Bu metod tahmini maliyeti hesaplar.
        
        Args:
            text: Sese çevrilecek metin
            
        Returns:
            Tahmini maliyet (USD)
        """
        # ElevenLabs fiyatlandırması (yaklaşık)
        # Free tier: 10,000 karakter/ay
        # Starter: $5/ay - 30,000 karakter
        # Pro: $22/ay - 100,000 karakter
        
        characters = len(text)
        
        # Pro plan varsayımıyla
        cost_per_1000_chars = 0.22  # $0.22 per 1000 karakter
        
        estimated_cost = (characters / 1000) * cost_per_1000_chars
        
        return round(estimated_cost, 4)
    
    def get_statistics(self) -> dict:
        """API kullanım istatistiklerini döndür"""
        avg_chars = (
            self.total_characters / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_characters": self.total_characters,
            "average_characters_per_request": round(avg_chars, 1),
            "estimated_total_cost": f"${self.calculate_cost('') * self.total_requests:.2f}",
            "voice_id": self.voice_id,
            "model": self.model_id
        }
    
    async def test_connection(self) -> bool:
        """
        API bağlantısını test et.
        
        Kısa bir test metni göndererek API'nin çalıştığını doğrular.
        """
        try:
            logger.info("ElevenLabs API bağlantısı test ediliyor...")
            
            # Test metni
            test_text = "Merhaba, bu bir test mesajıdır."
            
            # Küçük bir ses üret
            audio_data = await self.text_to_speech(
                text=test_text,
                output_format="mp3_44100_128",
                stream=False
            )
            
            # Ses verisi geldi mi kontrol et
            if audio_data and len(audio_data) > 1000:  # En az 1KB
                logger.info("✅ ElevenLabs API bağlantısı başarılı!")
                return True
            else:
                logger.warning("Ses verisi çok küçük veya boş")
                return False
                
        except Exception as e:
            logger.error(f"❌ ElevenLabs API bağlantı hatası: {str(e)}")
            return False


# Test için
if __name__ == "__main__":
    async def test():
        client = ElevenLabsClient()
        
        # Bağlantı testi
        if await client.test_connection():
            print("✅ ElevenLabs bağlantısı başarılı!")
            
            # Ses listesi
            voices = await client.get_voices()
            if voices:
                print(f"\nKullanılabilir sesler ({len(voices)} adet):")
                for voice in voices[:3]:  # İlk 3 ses
                    print(f"  - {voice['name']} ({voice['voice_id'][:8]}...)")
            
            # Maliyet hesaplama
            sample_text = "Bu örnek bir mülakat sorusudur. Deneyiminizden bahseder misiniz?"
            cost = client.calculate_cost(sample_text)
            print(f"\nÖrnek metin maliyeti: ${cost:.4f}")
            
            # İstatistikler
            print(f"\nİstatistikler: {client.get_statistics()}")
        else:
            print("❌ ElevenLabs bağlantısı başarısız!")
    
    asyncio.run(test())