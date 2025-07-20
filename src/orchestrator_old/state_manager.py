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
    recording_path: Optional[str] = None
    transcript_path: Optional[str] = None
    
    # Hata ve uyarılar
    errors: list[Dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    # İstatistikler
    statistics: Dict[str, Any] = Field(default_factory=dict)
    
    def add_qa(self, question: str, answer: str, timestamp: float):
        """Soru-cevap çifti ekle"""
        self.transcript.append(QA(role="ai", text=question, ts=timestamp))
        self.transcript.append(QA(role="human", text=answer, ts=timestamp + 0.1))
    
    def mark_phase_completed(self, phase: str):
        """Bir fazı tamamlanmış olarak işaretle"""
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
    
    def update_statistics(self, key: str, value: Any):
        """İstatistik güncelle"""
        self.statistics[key] = value
    
    def add_error(self, error_type: str, details: str, timestamp: Optional[datetime] = None):
        """Hata kaydet"""
        self.errors.append({
            "type": error_type,
            "details": details,
            "timestamp": (timestamp or datetime.now()).isoformat()
        })


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
        self.auto_save_interval = 30  # 30 saniyede bir otomatik kayıt
        self._save_task = None
        
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
            logger.info(f"Kaldığı yer: Faz={self.current_session.current_phase}, Soru={self.current_session.current_question_index}")
            
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
                json.dump(
                    self.current_session.model_dump(mode='json'),
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str  # datetime için
                )
            
            # Atomik taşıma
            temp_file.replace(final_file)
            
            logger.debug(f"Durum kaydedildi: {self.current_session.session_id}")
            
        except Exception as e:
            logger.error(f"Durum kaydetme hatası: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def update_phase(self, phase: str, question_index: int = 0):
        """Mevcut fazı güncelle"""
        if self.current_session:
            self.current_session.current_phase = phase
            self.current_session.current_question_index = question_index
            self.save_state()
    
    def update_transcript(self, question: str, answer: str):
        """Transcript'e yeni Q&A ekle"""
        if self.current_session:
            self.current_session.add_qa(
                question=question,
                answer=answer,
                timestamp=datetime.now().timestamp()
            )
            self.save_state()
    
    def mark_completed(self):
        """Mülakatı tamamlanmış olarak işaretle"""
        if self.current_session:
            self.current_session.state = InterviewState.COMPLETED
            self.current_session.total_duration = (
                datetime.now() - self.current_session.started_at
            ).total_seconds()
            self.save_state()
            self._stop_auto_save()
    
    def mark_failed(self, error: str):
        """Mülakatı başarısız olarak işaretle"""
        if self.current_session:
            self.current_session.state = InterviewState.FAILED
            self.current_session.add_error("fatal", error)
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
                logger.debug("Otomatik kayıt yapıldı")
    
    def _start_auto_save(self):
        """Otomatik kayıt görevini başlat"""
        if self._save_task:
            self._save_task.cancel()
        
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
                
                if data.get('state') == InterviewState.IN_PROGRESS.value:
                    recovery_sessions.append({
                        'session_id': data['session_id'],
                        'candidate_name': data['candidate_info']['name'],
                        'position': data['job_info']['title'],
                        'last_checkpoint': data['last_checkpoint'],
                        'current_phase': data['current_phase'],
                        'questions_completed': len(data['transcript']) // 2
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
    
    def export_session(self, session_id: str, output_path: Path):
        """
        Oturumu dışa aktar.
        
        Arşivleme veya analiz için oturum verilerini
        farklı bir konuma kopyalar.
        """
        state_file = self.state_dir / f"{session_id}.json"
        
        if not state_file.exists():
            raise FileNotFoundError(f"Oturum bulunamadı: {session_id}")
        
        # Çıktı dizinini oluştur
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dosyayı kopyala
        import shutil
        shutil.copy2(state_file, output_path)
        
        logger.info(f"Oturum dışa aktarıldı: {session_id} -> {output_path}")


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
    
    # Biraz veri ekle
    state_manager.update_transcript("Test question?", "Test answer.")
    state_manager.update_phase("technical", 1)
    
    # Kurtarılabilir oturumları listele
    recoverable = state_manager.get_recovery_info()
    print(f"\nKurtarılabilir oturumlar: {len(recoverable)}")
    for info in recoverable:
        print(f"  - {info['session_id']}: {info['candidate_name']} ({info['questions_completed']} soru)")