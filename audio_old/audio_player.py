"""
AudioPlayer – ElevenLabs TTS çıktısını gerçek-zamanlı oynatır.
Ses-video senkronizasyonu için timestamp desteği eklenmiştir.
"""

import asyncio
from pathlib import Path
from typing import Union, AsyncGenerator, Optional, Callable
import time

import pyaudio
from loguru import logger
import numpy as np


class AudioPlayer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_playing: bool = False
        self.volume: float = 1.0  # 0.0-1.0
        self.on_finish: Optional[Callable[[], None]] = None
        self.play_start_time: Optional[float] = None
        self.play_end_time: Optional[float] = None
        
    async def play(
        self,
        source: Union[bytes, AsyncGenerator[bytes, None], str, Path],
        sample_rate: int = 44100,
        channels: int = 1,
        sample_width: int = 2,
        chunk: int = 2048,
    ):
        """Kaynağı çalmaya başla (bytes, async-gen veya dosya)."""
        if self.is_playing:
            logger.warning("Başka bir ses hâlâ çalıyor")
            return
            
        self.is_playing = True
        self.play_start_time = time.time()
        
        try:
            self.stream = self.audio.open(
                format=self.audio.get_format_from_width(sample_width),
                channels=channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=chunk,
                stream_callback=None,
            )
            
            frame_count = 0
            
            async for frame in self._frame_iter(source, chunk):
                if not self.is_playing:
                    break
                    
                # Ses seviyesi (gain)
                if self.volume != 1.0:
                    frame = self._apply_gain(frame, self.volume)
                
                # Frame'i oynat
                self.stream.write(frame)
                frame_count += 1
                
                # CPU'ya nefes aldır ama çok bekleme (senkronizasyon için)
                if frame_count % 10 == 0:  # Her 10 frame'de bir
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Ses oynatma hatası: {e}")
        finally:
            self.stop()
            self.play_end_time = time.time()
            
            if self.play_start_time and self.play_end_time:
                duration = self.play_end_time - self.play_start_time
                logger.info(f"Ses oynatma süresi: {duration:.2f} saniye")
                
            if self.on_finish:
                self.on_finish()
    
    def stop(self):
        """Çalmayı durdur ve kaynakları bırak."""
        if not self.is_playing:
            return
            
        self.is_playing = False
        
        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
            if self.stream:
                self.stream.close()
        except Exception as e:
            logger.warning(f"Stream kapatma hatası: {e}")
        finally:
            self.stream = None
            logger.debug("AudioPlayer durduruldu")
    
    async def _frame_iter(
        self,
        src: Union[bytes, AsyncGenerator[bytes, None], str, Path],
        chunk: int,
    ) -> AsyncGenerator[bytes, None]:
        """Kaynağı parça parça getirir – türden bağımsız."""
        if isinstance(src, (bytes, bytearray)):
            # Bytes verisi için
            for i in range(0, len(src), chunk):
                yield src[i : i + chunk]
                
        elif isinstance(src, (str, Path)):
            # Dosya için
            path = Path(src)
            with path.open("rb") as f:
                while True:
                    data = f.read(chunk)
                    if not data:
                        break
                    yield data
                    
        else:  # async-generator
            async for data in src:
                yield data
    
    def _apply_gain(self, frame: bytes, gain: float) -> bytes:
        """Basit PCM16 ses seviyesi ayarı."""
        if not frame:
            return frame
            
        try:
            arr = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            arr *= gain
            arr = np.clip(arr, -32768, 32767).astype(np.int16)
            return arr.tobytes()
        except Exception as e:
            logger.warning(f"Gain uygulama hatası: {e}")
            return frame
    
    def get_play_duration(self) -> Optional[float]:
        """Son oynatmanın süresini döndür"""
        if self.play_start_time and self.play_end_time:
            return self.play_end_time - self.play_start_time
        return None
    
    def __del__(self):
        try:
            self.stop()
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except Exception:
            pass