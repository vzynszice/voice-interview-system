"""
Mülakat Durumu Yönetimi - State Persistence

Bu modül, mülakat sırasında sistemin durumunu sürekli olarak kaydeder.
Herhangi bir kesinti durumunda, mülakat kaldığı yerden devam edebilir.

Düşünün ki bir kitap okurken ayraç koyuyorsunuz - bu sistem de
mülakatın "ayracını" tutuyor.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from loguru import logger

from src.orchestrator.schema import JobInfo, CandidateInfo, QA


class InterviewState(str, Enum):
    """Mülakat durumları"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class InterviewSession(BaseModel):
    """
    Bir mülakat oturumunun tüm durumu.
    
    Bu model, mülakatın herhangi bir anındaki tam durumunu temsil eder.
    Bir fotoğraf gibi düşünün - o anki her şeyi kaydeder.
    """
    session_id: str = Field(..., description="Benzersiz oturum kimliği")
    state: InterviewState = Field(default=InterviewState.NOT_STARTED)
    
    # Mülakat bilgileri
    job_info: JobInfo
    candidate_info: CandidateInfo
    
    # İlerleme durumu
    current_phase: str = Field(default="warmup")
    current_question_index: int = Field(default=0)
    completed_phases: list[str] = Field(default_factory=list)
    
    # Konuşma geçmişi
    transcript: list[QA] = Field(default_factory=list)
    
    # Zaman bilgileri
    started_at: Optional[datetime] = None
    last_checkpoint: datetime = Field(default_factory=datetime.now)
    total_duration: float = Field(default=0.0, description="Toplam süre (saniye)")
    
    # Kayıt bilgileri
    transcript_path: Optional[str] = None
    
    # Hata ve uyarılar
    errors: list[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        use_enum_values = True

class StateManager:
    """
    Mülakat durumunu yöneten sınıf.
    
    Bu sınıf, mülakatın durumunu düzenli olarak diske kaydeder
    ve gerektiğinde geri yükler. Bir oyundaki "save game" özelliği
    gibi düşünebilirsiniz.
    """
    
    def __init__(self, state_dir: Path = Path("data/state")):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[InterviewSession] = None
        self.auto_save_interval = 60  # 30 saniyede bir otomatik kayıt
        self._save_task: Optional[asyncio.Task] = None
        logger.info(f"State Manager başlatıldı. Dizin: {self.state_dir}")
    
    def create_session(self, job: JobInfo, candidate: CandidateInfo) -> InterviewSession:
        """
        Yeni bir mülakat oturumu oluştur.
        
        Her mülakat için benzersiz bir kimlik üretir ve
        başlangıç durumunu ayarlar.
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = InterviewSession(
            session_id=session_id,
            job_info=job,
            candidate_info=candidate,
            started_at=datetime.now()
        )
        
        # İlk kaydı yap
        self.save_state()
        
        # Otomatik kayıt görevini başlat
        self._start_auto_save()
        
        logger.info(f"Yeni oturum oluşturuldu: {session_id}")
        return self.current_session
    
    def load_session(self, session_id: str) -> Optional[InterviewSession]:
        """
        Önceki bir oturumu yükle.
        
        Kesintiye uğramış bir mülakatı kaldığı yerden
        devam ettirmek için kullanılır.
        """
        state_file = self.state_dir / f"{session_id}.json"
        
        if not state_file.exists():
            logger.warning(f"Oturum dosyası bulunamadı: {session_id}")
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_session = InterviewSession(**data)
            
            # Otomatik kayıt görevini başlat
            self._start_auto_save()
            
            logger.info(f"Oturum yüklendi: {session_id}")
            return self.current_session
            
        except Exception as e:
            logger.error(f"Oturum yükleme hatası: {e}")
            return None
    
    def save_state(self):
        """
        Mevcut durumu diske kaydet.
        
        Bu metod, mülakatın tam durumunu JSON formatında kaydeder.
        Güvenlik için önce geçici dosyaya yazar, sonra taşır.
        """
        if not self.current_session:
            return
        
        # Checkpoint zamanını güncelle
        self.current_session.last_checkpoint = datetime.now()
        
        # Geçici dosyaya yaz (atomik işlem için)
        temp_file = self.state_dir / f"{self.current_session.session_id}.tmp"
        final_file = self.state_dir / f"{self.current_session.session_id}.json"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                # Pydantic'in model_dump_json metodu daha güvenilirdir.
                f.write(self.current_session.model_dump_json(indent=2))
            temp_file.replace(final_file)
            logger.debug(f"Durum kaydedildi: {self.current_session.session_id}")
        except Exception as e:
            logger.error(f"Durum kaydetme hatası: {e}")
    
    def mark_completed(self):
        """Mülakatı tamamlanmış olarak işaretle"""
        if self.current_session:
            self.current_session.state = InterviewState.COMPLETED
            self.save_state()
            self._stop_auto_save()
    
    def mark_failed(self, error: str):
        """Mülakatı başarısız olarak işaretle"""
        if self.current_session:
            self.current_session.state = InterviewState.FAILED
            self.current_session.errors.append({"type": "fatal", "details": error})
            self.save_state()
            self._stop_auto_save()
    
    async def _auto_save_loop(self):
        """
        Otomatik kayıt döngüsü.
        
        Belirli aralıklarla durumu otomatik olarak kaydeder.
        Elektrik kesintisi gibi durumlara karşı koruma sağlar.
        """
        while True:
            await asyncio.sleep(self.auto_save_interval)
            if self.current_session and self.current_session.state == InterviewState.IN_PROGRESS:
                self.save_state()
    
    def _start_auto_save(self):
        """Otomatik kayıt görevini başlat"""
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._auto_save_loop())
    
    def _stop_auto_save(self):
        """Otomatik kayıt görevini durdur"""
        if self._save_task:
            self._save_task.cancel()
            self._save_task = None
    
    def get_recovery_info(self) -> list[Dict[str, Any]]:
        """
        Kurtarılabilir oturumları listele.
        
        Yarıda kalmış mülakatları bulur ve bilgilerini döndürür.
        """
        recovery_sessions = []
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get('state') in [InterviewState.IN_PROGRESS.value, InterviewState.PAUSED.value]:
                    recovery_sessions.append({
                        'session_id': data['session_id'],
                        'candidate_name': data['candidate_info']['name'],
                        'position': data['job_info']['title'],
                    })
                    
            except Exception as e:
                logger.warning(f"Oturum dosyası okunamadı: {state_file}, Hata: {e}")
        
        return recovery_sessions
    
    def cleanup_old_sessions(self, days: int = 7):
        """
        Eski oturum dosyalarını temizle.
        
        Belirtilen günden eski tamamlanmış oturumları siler.
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        cleaned_count = 0
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                # Dosya yaşını kontrol et
                if state_file.stat().st_mtime < cutoff_time:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Sadece tamamlanmış oturumları sil
                    if data.get('state') in [InterviewState.COMPLETED.value, InterviewState.FAILED.value]:
                        state_file.unlink()
                        cleaned_count += 1
                        
            except Exception as e:
                logger.warning(f"Temizlik hatası: {state_file}, {e}")
        
        if cleaned_count > 0:
            logger.info(f"{cleaned_count} eski oturum dosyası temizlendi")
    

# Global state manager instance
state_manager = StateManager()


# Test için
if __name__ == "__main__":
    # Test verileri
    job = JobInfo(
        title="Python Developer",
        company="Test Corp",
        requirements={"technical_skills": ["Python", "Django"]}
    )
    
    candidate = CandidateInfo(
        name="Test User",
        current_position="Developer",
        years_experience=3,
        key_skills=["Python"]
    )
    
    # Yeni oturum oluştur
    session = state_manager.create_session(job, candidate)
    print(f"Oturum oluşturuldu: {session.session_id}")
    
    # Kurtarılabilir oturumları listele
    recoverable = state_manager.get_recovery_info()
    print(f"\nKurtarılabilir oturumlar: {len(recoverable)}")
    for info in recoverable:
        print(f"  - {info['session_id']}: {info['candidate_name']} ({info['questions_completed']} soru)")