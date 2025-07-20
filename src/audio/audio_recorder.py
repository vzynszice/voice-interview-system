# src/audio/audio_recorder.py

import asyncio
import io
import queue
import time
import wave
from typing import Optional, Callable, AsyncGenerator

import numpy as np
import pyaudio
import webrtcvad
from loguru import logger

from config import config

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        self.format = pyaudio.paInt16
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(config.audio.vad_mode)
        self.silence_threshold = config.audio.silence_threshold
        self.silence_duration = config.audio.silence_duration
        self.is_recording = False
        self.stream = None
        self.frames = []
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_audio_chunk = None
        self.total_recordings = 0
        self.total_duration = 0.0
        self.input_device_index = next(
            (i for i in range(self.audio.get_device_count())
            if self.audio.get_device_info_by_index(i)['maxInputChannels'] > 0),
            None
        )
        logger.info(f"Kullanılacak giriş cihazı: {self.input_device_index}")
        logger.info("Ses kaydedici başlatıldı")

    def _list_audio_devices(self):
        logger.info("Mevcut ses cihazları:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                logger.info(f"  Giriş #{i}: {info['name']} ({info['maxInputChannels']} kanal)")

    async def start_recording(
        self,
        max_duration: Optional[float] = None,
        auto_stop_on_silence: bool = True
    ) -> bytes:
        if self.is_recording:
            logger.warning("Kayıt zaten devam ediyor")
            return b''
        logger.info("Ses kaydı başlatılıyor...")
        self.is_recording = True
        self.frames = []
        start_time = time.time()
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
            )
            if self.on_speech_start: await self.on_speech_start()
            silence_start = None
            while self.is_recording:
                if max_duration and (time.time() - start_time) > max_duration:
                    logger.info("Maksimum kayıt süresi aşıldı")
                    break
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    if not data: continue
                    self.frames.append(data)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    volume = np.sqrt(np.mean(audio_array**2))
                    if auto_stop_on_silence:
                        is_speech = self._is_speech(data)
                        if is_speech:
                            silence_start = None
                            if self.on_audio_chunk: await self.on_audio_chunk(data, volume)
                        else:
                            if silence_start is None: silence_start = time.time()
                            elif time.time() - silence_start > self.silence_duration:
                                logger.info("Sessizlik algılandı, kayıt durduruluyor")
                                break
                except Exception as e:
                    logger.error(f"Ses okuma hatası: {e}")
                    break
                await asyncio.sleep(0.01)
            self.stop_recording()
            duration = time.time() - start_time
            self.total_duration += duration
            self.total_recordings += 1
            if self.on_speech_end: await self.on_speech_end()
            wav_data = self._frames_to_wav(self.frames)
            logger.info(f"Kayıt tamamlandı. Süre: {duration:.1f}s, Boyut: {len(wav_data)/1024:.1f}KB")
            return wav_data
        except Exception as e:
            logger.error(f"Kayıt hatası: {e}")
            self.stop_recording()
            raise
    async def listen_for_interruption(self) -> bool:
        """
        Sadece konuşma başlayana kadar dinler. Konuşma algılandığı an döner.
        Akustik geri beslemeyi (feedback) önlemek için birkaç ardışık
        ses parçası algılaması gerekir.
        """
        if self.is_recording:
            logger.warning("Zaten bir kayıt işlemi var, kesinti dinlenemiyor.")
            return False

        stream = None
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
            )
            logger.debug("🎤 Kesinti için dinleme başladı...")

            # Art arda kaç tane sesli chunk yakaladığımızı sayacağız
            speech_chunks_count = 0
            # Tetiklenmek için gereken ardışık chunk sayısı
            required_chunks = 3  # Bu değeri artırarak hassasiyeti azaltabilirsiniz

            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)

                if self._simple_volume_check(data):
                    speech_chunks_count += 1
                else:
                    # Sessizlik varsa sayacı sıfırla
                    speech_chunks_count = 0

                # Eğer yeterli sayıda ardışık sesli chunk yakaladıysak, bu bir kesintidir.
                if speech_chunks_count >= required_chunks:
                    logger.info(f"✅ Kesinti algılandı! ({required_chunks} ardışık sesli chunk yakalandı).")
                    return True

                # Ana task'tan iptal sinyali gelirse döngüyü kırmak için
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.debug("Kesinti dinleme görevi iptal edildi.")
            return False
        except Exception as e:
            logger.error(f"Kesinti dinleme sırasında hata: {e}")
            return False
        finally:
            if stream:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            logger.debug("Kesinti dinleme stream'i kapatıldı.")

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            try:
                if self.stream.is_active(): self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Stream kapatılırken hata oluştu: {str(e)}")
            finally:
                self.stream = None
        logger.debug("Kayıt durduruldu")

    def _is_speech(self, audio_chunk: bytes) -> bool:
        if not audio_chunk: return False
        try:
            chunk_duration_ms = 20
            chunk_size_bytes = int(self.sample_rate * chunk_duration_ms / 1000) * 2
            if len(audio_chunk) >= chunk_size_bytes:
                return self.vad.is_speech(audio_chunk[:chunk_size_bytes], self.sample_rate)
            else:
                return self._simple_volume_check(audio_chunk)
        except Exception as e:
            logger.debug(f"VAD hatası: {e}")
            return self._simple_volume_check(audio_chunk)

    def _simple_volume_check(self, audio_chunk: bytes) -> bool:
        if not audio_chunk: return False
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        if audio_array.size == 0: return False # Ek bir kontrol
        volume = np.sqrt(np.mean(audio_array.astype(np.float64)**2))
        return volume > self.silence_threshold

    def _frames_to_wav(self, frames: list) -> bytes:
        if not frames: return b''
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        buffer.seek(0)
        return buffer.read()

    # === EKSİK METOTLAR BURAYA GERİ EKLENDİ ===

    def get_statistics(self) -> dict:
        """Kayıt istatistiklerini döndür"""
        avg_duration = (self.total_duration / self.total_recordings if self.total_recordings > 0 else 0)
        return {
            "total_recordings": self.total_recordings,
            "total_duration": f"{self.total_duration:.1f} saniye",
            "average_duration": f"{avg_duration:.1f} saniye",
            "sample_rate": f"{self.sample_rate} Hz",
            "silence_threshold": self.silence_threshold
        }

    def calibrate_silence_threshold(self, duration: float = 3.0) -> int:
        """Ortam gürültüsüne göre sessizlik eşiğini ayarla."""
        logger.info(f"Sessizlik eşiği kalibrasyonu başlatılıyor ({duration}s)...")
        volumes = []
        stream = None
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size
            )
            start_time = time.time()
            while time.time() - start_time < duration:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_array.astype(np.float64)**2))
                volumes.append(volume)
            
            if stream.is_active(): stream.stop_stream()
            stream.close()

            if not volumes: return self.silence_threshold

            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            threshold = int(mean_volume + 2 * std_volume)
            logger.info(f"Kalibrasyon tamamlandı. Önerilen eşik: {threshold}")
            return threshold
        except Exception as e:
            logger.error(f"Kalibrasyon hatası: {e}")
            if stream:
                if stream.is_active(): stream.stop_stream()
                stream.close()
            return self.silence_threshold

    def cleanup(self):
        """Kaynakları temizle"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
        logger.info("Ses kaydedici temizlendi")

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass