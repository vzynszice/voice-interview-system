"""
Sesli Mülakat Sistemi - Merkezi Konfigürasyon

Bu dosya, tüm sistem ayarlarını merkezi olarak yönetir. Çevre değişkenlerini
okur, varsayılan değerler sağlar ve tüm modüllerin kullanacağı ayarları sunar.

Tasarım Prensibi: Tek Gerçek Kaynağı (Single Source of Truth)
Tüm ayarlar buradan okunur, böylece değişiklik yapmak kolaylaşır.
"""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import Field,field_validator
from pydantic_settings import BaseSettings 

# .env dosyasını yükle
load_dotenv()


class APIConfig(BaseSettings):
    """API bağlantı ayarları (Google Cloud Translation v3 dâhil)"""

    # --- API Anahtarları / Kimlikler ---
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    elevenlabs_api_key: str = Field(..., env="ELEVENLABS_API_KEY")

    # Google Cloud Translation
    google_application_credentials: Path = Field(..., env="GOOGLE_APPLICATION_CREDENTIALS")
    google_cloud_project_id: Optional[str] = Field(None, env="GOOGLE_CLOUD_PROJECT_ID")
    gct_location: str = Field("global", env="GCT_LOCATION")
    gct_source_lang: str = Field("tr", env="GCT_SOURCE_LANG")
    gct_target_lang: str = Field("en", env="GCT_TARGET_LANG")

    # --- Model Ayarları ---
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    elevenlabs_voice_id: str = Field("21m00Tcm4TlvDq8ikWAM", env="ELEVENLABS_VOICE_ID")

    # --- Doğrulamalar ---
    @field_validator("groq_api_key", "openai_api_key", "elevenlabs_api_key",mode="before")
    @classmethod
    def _nonempty(cls, v: str, info):
        if not v or v.startswith("your_"):
            raise ValueError(f"{info.field_name} ayarlanmamış. .env kontrol edin.")
        return v

    @field_validator("google_application_credentials", mode="before")
    @classmethod
    def _path_exists(cls, v: Path):
        v = Path(v).expanduser()
        if not v.exists():
            raise ValueError(f"Google credentials bulunamadı: {v}")
        return v

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


class AudioConfig(BaseSettings):
    """Ses işleme ayarları"""

    # Ses Parametreleri
    sample_rate: int = Field(16000, env="AUDIO_SAMPLE_RATE")
    channels: int = Field(1, env="AUDIO_CHANNELS")
    chunk_size: int = Field(1024, env="AUDIO_CHUNK_SIZE")

    # Sessizlik Algılama
    silence_threshold: int = Field(500, env="SILENCE_THRESHOLD")
    silence_duration: float = Field(2.0, env="SILENCE_DURATION")

    # Ses Formatı
    audio_format: str = "wav"
    audio_encoding: str = "pcm16"

    @field_validator("sample_rate", mode="before")
    @classmethod
    def _valid_sr(cls, v: int):  # type: ignore[override]
        if v not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError("Geçersiz sample‑rate")
        return v


class RecordingConfig(BaseSettings):
    """Ekran / video kayıt ayarları"""

    video_fps: int = Field(30, env="VIDEO_FPS")
    video_codec: str = Field("mp4v", env="VIDEO_CODEC")
    recording_output_dir: Path = Field(Path("./data/recordings"), env="RECORDING_OUTPUT_DIR")

    video_format: str = "mp4"
    include_audio: bool = True
    screen_capture_area: Optional[tuple] = None  # None = tam ekran

    @field_validator("recording_output_dir", mode="before")
    @classmethod
    def _mkdir(cls, v: Path):  # type: ignore[override]
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ApplicationConfig(BaseSettings):
    """Uygulama genel ayarları"""

    base_dir: Path = Path(__file__).parent
    transcript_dir: Path = Field(Path("./data/transcripts"), env="TRANSCRIPT_DIR")
    generated_data_dir: Path = Field(Path("./data/generated"), env="GENERATED_DATA_DIR")

    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    max_interview_duration: int = Field(45, env="MAX_INTERVIEW_DURATION")  # dakika
    max_response_time: int = Field(300, env="MAX_RESPONSE_TIME")  # saniye

    debug: bool = Field(False, env="DEBUG")
    test_mode: bool = Field(False, env="TEST_MODE")

    @field_validator("transcript_dir", "generated_data_dir", mode="before")
    @classmethod
    def _mkdirs(cls, v: Path):  # type: ignore[override]
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    def setup_logging(self):
        logger.remove()
        logger.add(sys.stderr, format=self.log_format, level=self.log_level, colorize=True)
        log_file = self.base_dir / "logs" / "interview_{time}.log"
        logger.add(log_file, format=self.log_format, level="DEBUG", rotation="1 day", retention="7 days", compression="zip")
        logger.info("Loglama başlatıldı")


class InterviewConfig(BaseSettings):
    """Mülakat akışı ayarları"""

    interview_phases: list[str] = [
        "warmup",
        "technical",
        "behavioral",
        "situational",
        "closing",
    ]

    questions_per_phase: dict[str, int] = {
        "warmup": 1,
        "technical": 2,
        "behavioral": 1,
        "situational": 1,
        "closing": 1,
    }

    system_prompt_template: str = (
        """You are an experienced HR professional conducting a job interview.\n"
        "You are interviewing for the position of {position} at {company}.\n"
        "The candidate's name is {candidate_name}.\n\n"
        "Important instructions:\n"
        "1. Ask ONE question at a time\n"
        "2. Keep questions relevant to the job requirements\n"
        "3. Be professional but friendly\n"
        "4. Listen actively to responses\n"
        "5. Ask follow-up questions when appropriate\n\n"
        "Job Requirements:\n{job_requirements}\n\n"
        "Candidate Background:\n{candidate_summary}\n"""
    )

    evaluation_weights: dict[str, float] = {
        "technical_competency": 0.35,
        "communication_skills": 0.25,
        "problem_solving": 0.20,
        "cultural_fit": 0.20,
    }


class Config:
    """Singleton ana konfigürasyon nesnesi"""

    _instance: Optional["Config"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.api = APIConfig()
        self.audio = AudioConfig()
        self.recording = RecordingConfig()
        self.app = ApplicationConfig()
        self.interview = InterviewConfig()
        self.app.setup_logging()
        self._initialized = True

    # ---------------------------------------------------------------------
    # Yardımcı Metotlar
    # ---------------------------------------------------------------------
    def validate(self) -> bool:
        """Tüm kritik ayarların mevcut ve geçerli olduğunu kontrol et"""
        try:
            assert self.api.groq_api_key
            assert self.api.openai_api_key
            assert self.api.elevenlabs_api_key
            assert self.api.google_application_credentials.exists()

            assert self.app.transcript_dir.exists()
            assert self.app.generated_data_dir.exists()
            assert self.recording.recording_output_dir.exists()

            logger.info("Konfigürasyon doğrulaması başarılı")
            return True
        except AssertionError as e:
            logger.error(f"Konfigürasyon doğrulama hatası: {e}")
            return False

    def get_summary(self) -> dict:
        return {
        "api_keys_loaded": all([
            self.api.groq_api_key,
            self.api.openai_api_key,
            self.api.elevenlabs_api_key
        ]),
        "audio_settings": {
            "sample_rate": self.audio.sample_rate,
            "channels": self.audio.channels
        },
        "recording_settings": {
            "fps": self.recording.video_fps,
            "output_dir": str(self.recording.recording_output_dir)
        },
        "interview_settings": {
            "total_questions": sum(self.interview.questions_per_phase.values()),
            "max_duration": self.app.max_interview_duration
        }
    }


# Global config instance
config = Config()


# ------------------------------------------------------------
# Modülü doğrudan çalıştırırsanız küçük bir sağlık testi yapar
# ------------------------------------------------------------
if __name__ == "__main__":
    import json
    print(json.dumps(config.get_summary(), indent=2))
