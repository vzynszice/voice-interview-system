"""
Sesli Mülakat Sistemi - Lokal Konfigürasyon

Bu dosya, tamamen lokal çalışan sistem için ayarları yönetir.
API anahtarlarına ihtiyaç yoktur, tüm modeller lokalde çalışır.
"""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from loguru import logger
from src.utils.logger import setup_logging
load_dotenv()


class LocalModelConfig(BaseSettings):
    """Lokal model ayarları"""
    
    # Ollama ayarları
    ollama_host: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field("gemma3:1b", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(30, env="OLLAMA_TIMEOUT")
    
    # Whisper ayarları
    whisper_model: str = Field("deepdml/faster-whisper-large-v3-turbo-ct2", env="WHISPER_MODEL")
    whisper_device: str = Field("cpu", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", env="WHISPER_COMPUTE")
    whisper_language: str = Field("tr", env="WHISPER_LANGUAGE")
    
    # TTS ayarları
    tts_backend: str = Field("pyttsx3", env="TTS_BACKEND")
    tts_rate: int = Field(175, env="TTS_RATE")
    
    # Model dizinleri
    models_dir: Path = Field(Path("./models"), env="MODELS_DIR")
    
    @field_validator("models_dir", mode="before")
    @classmethod
    def create_models_dir(cls, v: Path) -> Path:
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


class AudioConfig(BaseSettings):
    """Ses işleme ayarları"""
    
    # Ses Parametreleri
    sample_rate: int = Field(48000, env="AUDIO_SAMPLE_RATE")
    channels: int = Field(1, env="AUDIO_CHANNELS")
    chunk_size: int = Field(960, env="AUDIO_CHUNK_SIZE")
    
    # Sessizlik Algılama
    silence_threshold: int = Field(500, env="SILENCE_THRESHOLD")
    silence_duration: float = Field(2.0, env="SILENCE_DURATION")
    
    # Voice Activation Detection (VAD)
    vad_mode: int = Field(2, env="VAD_MODE")  # 0-3 arası, 3 en agresif
    vad_frame_duration: int = Field(20, env="VAD_FRAME_DURATION")  # ms
    
    @field_validator("sample_rate", mode="before")
    @classmethod
    def valid_sample_rate(cls, v: int) -> int:
        if v not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError("Geçersiz sample rate")
        return v
    
    @field_validator("vad_mode", mode="before")
    @classmethod
    def valid_vad_mode(cls, v: int) -> int:
        if v not in [0, 1, 2, 3]:
            raise ValueError("VAD modu 0-3 arasında olmalı")
        return v


class ApplicationConfig(BaseSettings):
    """Uygulama genel ayarları"""
    
    base_dir: Path = Path(__file__).parent.parent
    transcript_dir: Path = Field(Path("./data/transcripts"), env="TRANSCRIPT_DIR")
    generated_data_dir: Path = Field(Path("./data/generated"), env="GENERATED_DATA_DIR")
    temp_dir: Path = Field(Path("./temp"), env="TEMP_DIR")
    
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    max_interview_duration: int = Field(45, env="MAX_INTERVIEW_DURATION")  # dakika
    max_response_time: int = Field(300, env="MAX_RESPONSE_TIME")  # saniye
    
    # Streaming ayarları
    enable_streaming_tts: bool = Field(True, env="ENABLE_STREAMING_TTS")
    max_parallel_tts: int = Field(3, env="MAX_PARALLEL_TTS")
    
    # Voice activation ayarları
    enable_voice_activation: bool = Field(False, env="ENABLE_VOICE_ACTIVATION")
    
    debug: bool = Field(False, env="DEBUG")
    test_mode: bool = Field(False, env="TEST_MODE")
    
    @field_validator("transcript_dir", "generated_data_dir", "temp_dir", mode="before")
    @classmethod
    def create_dirs(cls, v: Path) -> Path:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


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
    
    # Çeviri ayarları (opsiyonel)
    enable_translation: bool = Field(True, env="ENABLE_TRANSLATION")
    translation_backend: str = Field("none", env="TRANSLATION_BACKEND")  # none, google, local
    
    system_prompt_template: str = """You are an experienced HR professional conducting a job interview.
You are interviewing for the position of {position} at {company}.
The candidate's name is {candidate_name}.

Important instructions:
1. Ask ONE question at a time
2. Keep questions relevant to the job requirements  
3. Be professional but friendly
4. Listen actively to responses
5. Ask follow-up questions when appropriate

Job Requirements:
{job_requirements}

Candidate Background:
{candidate_summary}"""


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
            
        self.models = LocalModelConfig()
        self.audio = AudioConfig()
        self.app = ApplicationConfig()
        self.interview = InterviewConfig()
        
        # Logger'ı başlat
        setup_logging(log_level=self.app.log_level, base_dir=self.app.base_dir)
        
        self._initialized = True
    
    def validate(self) -> bool:
        """Lokal sistemin hazır olduğunu kontrol et"""
        try:
            # Ollama kontrolü
            import requests
            response = requests.get(f"{self.models.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error("Ollama servisi çalışmıyor!")
                return False
            
            # Model kontrolü
            models = [m["name"] for m in response.json().get("models", [])]
            if self.models.ollama_model not in models:
                logger.error(f"{self.models.ollama_model} modeli yüklü değil!")
                logger.info(f"Yüklemek için: ollama pull {self.models.ollama_model}")
                return False
            
            # Dizin kontrolleri
            assert self.app.transcript_dir.exists()
            assert self.app.generated_data_dir.exists()
            assert self.app.temp_dir.exists()
            assert self.models.models_dir.exists()
            
            logger.info("Lokal sistem doğrulaması başarılı")
            return True
            
        except Exception as e:
            logger.error(f"Sistem doğrulama hatası: {e}")
            return False
    
    def get_summary(self) -> dict:
        """Konfigürasyon özetini döndür"""
        return {
            "system_type": "Lokal",
            "models": {
                "llm": f"{self.models.ollama_model} (Ollama)",
                "stt": f"Whisper {self.models.whisper_model}",
                "tts": self.models.tts_backend,
                "translation": "Devre dışı" if not self.interview.enable_translation else self.interview.translation_backend
            },
            "audio_settings": {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "vad_enabled": self.app.enable_voice_activation,
                "streaming_tts": self.app.enable_streaming_tts
            },
            "interview_settings": {
                "total_questions": sum(self.interview.questions_per_phase.values()),
                "max_duration": self.app.max_interview_duration,
                "phases": len(self.interview.interview_phases)
            }
        }


# Global config instance
config = Config()


# Test için
if __name__ == "__main__":
    import json
    
    print("Lokal Sistem Konfigürasyonu")
    print("=" * 50)
    
    # Özet göster
    print(json.dumps(config.get_summary(), indent=2))
    
    # Doğrulama
    print("\nSistem Doğrulaması:")
    if config.validate():
        print("✅ Sistem kullanıma hazır!")
    else:
        print("❌ Sistem hazır değil, yukarıdaki hataları düzeltin.")