import asyncio
from enum import Enum
from functools import wraps
from typing import Coroutine, Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Hata ve Durum Sınıfları ---

class ErrorSeverity(str, Enum):
    """Hatanın ciddiyet seviyesi."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FATAL = "fatal"

class InterviewError(Exception):
    """Mülakat sistemine özel hata sınıfı."""
    def __init__(self, message, severity=ErrorSeverity.MEDIUM, recoverable=False, recovery_suggestion=""):
        super().__init__(message)
        self.severity = severity
        self.recoverable = recoverable
        self.recovery_suggestion = recovery_suggestion
    
    def __str__(self):
        return f"[{self.severity.value.upper()}] {super().__str__()}"

# --- Decorator'lar ve Yardımcı Fonksiyonlar ---

def api_retry_handler():
    """
    API çağrıları için otomatik yeniden deneme sağlayan bir decorator.
    Tenacity'yi doğru şekilde kullanır.
    """
    def decorator(func: Coroutine) -> Coroutine:
        @wraps(func)
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True  # 3 denemeden sonra hala hata varsa, hatayı yeniden fırlat
        )
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API çağrısı yeniden deneniyor: {func.__name__}, Hata: {str(e)}")
                raise
        return wrapper
    return decorator

def timeout_handler(seconds: int, message: str):
    """Bir fonksiyonun belirli bir sürede bitmesini zorlar."""
    def decorator(func: Coroutine) -> Coroutine:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise InterviewError(f"{message} ({seconds}s zaman aşımı)", severity=ErrorSeverity.HIGH)
        return wrapper
    return decorator

# --- Diğer Handler'lar (Gelecekte Doldurulabilir) ---

def safe_async_call(fallback_value: Any = None):
    def decorator(func: Coroutine) -> Coroutine:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Güvenli çağrı hatası ({func.__name__}): {e}")
                return fallback_value
        return wrapper
    return decorator

def rate_limit_handler(func): return func
def interview_phase_handler(func): return func
def audio_operation_handler(func): return func
def global_error_handler(func): return func