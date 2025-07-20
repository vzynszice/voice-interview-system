# screen_recorder.py - Düzeltilmiş ve İyileştirilmiş Versiyon

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Callable
import time

import cv2
import mss
import numpy as np
from loguru import logger

from config import config


class ScreenRecorder:
    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        fps: int = config.recording.video_fps,
        mic: bool = False,
    ):
        self.region = region
        self.fps = fps
        self.is_recording = False
        self._temp_video_path: Optional[Path] = None
        self._frame_count = 0
        self._start_time = None
        
    def get_resolution(self) -> Tuple[int, int]:
        with mss.mss() as sct:
            # Genellikle ana monitör `monitors[1]` olur.
            monitor = sct.monitors[1]
            if self.region:
                monitor = {
                    "left": self.region[0], 
                    "top": self.region[1], 
                    "width": self.region[2], 
                    "height": self.region[3]
                }
            
            # Kodek uyumluluğu için çözünürlüğün 2'ye bölünebilir olmasını sağla
            width = monitor["width"] - (monitor["width"] % 2)
            height = monitor["height"] - (monitor["height"] % 2)
            return width, height
    
    async def record(
        self,
        output_path: Path,
        max_duration: Optional[int] = None,
    ):
        """Ekran kaydını başlat."""
        self.is_recording = True
        self._temp_video_path = output_path
        self._temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._start_time = time.time()
        
        resolution = self.get_resolution()
        logger.info(f"📹 Ekran kaydı başlatılıyor: {resolution[0]}x{resolution[1]} @ {self.fps}fps")
        
        # Video codec ve ayarları
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(self._temp_video_path),
            fourcc,
            self.fps,
            resolution,
            isColor=True
        )
        
        if not writer.isOpened():
            logger.error("Video writer açılamadı!")
            return
        
        with mss.mss() as sct:
            if self.region:
                monitor = {
                    "left": self.region[0], 
                    "top": self.region[1], 
                    "width": self.region[2], 
                    "height": self.region[3]
                }
            else:
                monitor = sct.monitors[1]
            
            try:
                frame_duration = 1.0 / self.fps
                next_frame_time = time.time()
                
                while self.is_recording:
                    current_time = time.time()
                    
                    # Max duration kontrolü
                    if max_duration and (current_time - self._start_time) > max_duration:
                        logger.info(f"⏱️ Maksimum süre ({max_duration}s) doldu, kayıt durduruluyor")
                        break
                    
                    # Frame zamanlaması - sadece gerektiğinde frame al
                    if current_time >= next_frame_time:
                        # Ekran görüntüsü al
                        img = np.array(sct.grab(monitor))
                        
                        # BGRA'dan BGR'ye çevir (OpenCV formatı)
                        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        
                        # Çözünürlük uyumluluğu için yeniden boyutlandır
                        if frame.shape[:2][::-1] != resolution:
                            frame = cv2.resize(frame, resolution)
                        
                        # Frame'i yaz
                        writer.write(frame)
                        self._frame_count += 1
                        
                        # Sonraki frame zamanını ayarla
                        next_frame_time += frame_duration
                        
                        # Progress log (her 100 frame'de bir)
                        if self._frame_count % 100 == 0:
                            elapsed = current_time - self._start_time
                            logger.debug(f"Kayıt devam ediyor: {self._frame_count} frame, {elapsed:.1f}s")
                    
                    # CPU kullanımını azaltmak için kısa bekleme
                    # Ama çok uzun bekleme yapma ki frame kaçırma
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Kayıt hatası: {e}")
                
            finally:
                # Writer'ı düzgün kapat
                logger.info(f"🎬 Kayıt sonlandırılıyor... (Toplam {self._frame_count} frame)")
                
                # Buffer'ın boşalması için kısa bekleme
                await asyncio.sleep(0.5)
                
                # Video writer'ı kapat
                writer.release()
                cv2.destroyAllWindows()
                
                # İstatistikleri logla
                if self._start_time:
                    total_duration = time.time() - self._start_time
                    actual_fps = self._frame_count / total_duration if total_duration > 0 else 0
                    logger.info(
                        f"📊 Kayıt tamamlandı: {self._temp_video_path}\n"
                        f"   Süre: {total_duration:.1f}s\n"
                        f"   Frame sayısı: {self._frame_count}\n"
                        f"   Gerçek FPS: {actual_fps:.1f}"
                    )
                
                self.is_recording = False
    
    def stop(self):
        """Kaydı durdur."""
        logger.info("🛑 Kayıt durdurma sinyali gönderildi")
        self.is_recording = False
    
    def get_temp_video_path(self) -> Optional[Path]:
        """Kaydedilen geçici video dosyasının yolunu döndürür."""
        return self._temp_video_path
