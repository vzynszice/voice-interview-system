"""
Merkezi Loglama Sistemi
⋯ (üstteki açıklama metni aynı) ⋯
"""

import sys
import json
import asyncio
import functools
import time
from pathlib import Path
from loguru import logger

from config import config


class InterviewLogger:
    """
    Mülakat sistemi için özelleştirilmiş logger (Singleton)
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.setup_logger()
            self._initialized = True

    # ──────────────────────────────── 1️⃣  Ana kurulum ────────────────────────────────
    def setup_logger(self):
        logger.remove()  # eski handler’ları temizle

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Konsol (renkli)
        logger.add(
            sys.stderr,
            format=self._console_format,        # ← callable, içinde {timestamp} YOK
            level=config.app.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # Dosya: düz metin
        logger.add(
            log_dir / "interview_{time:YYYY-MM-DD}.log",
            format=self._file_format,
            level="DEBUG",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
        )

        # Dosya: hatalar
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format=self._file_format,
            level="ERROR",
            rotation="1 week",
            retention="3 months",
            encoding="utf-8",
        )

        # Dosya: JSON (serialize=True kendi JSON’unu üretir, format⟂)
        logger.add(
            log_dir / "structured_{time:YYYY-MM-DD}.json",
            level="INFO",
            rotation="1 day",
            retention="7 days",
            serialize=True,
            encoding="utf-8",
        )

        # Dosya: performans
        logger.add(
            log_dir / "performance_{time:YYYY-MM-DD}.log",
            format=self._performance_format,
            level="INFO",
            filter=lambda rec: rec["extra"].get("performance"),
            rotation="1 day",
            retention="14 days",
            encoding="utf-8",
        )

        logger.success("Loglama sistemi başarıyla yapılandırıldı")

    # ──────────────────────────────── 2️⃣  Format yardımcıları ─────────────────────────
    def _console_format(self, record):
        """Handler-callable: renkli, {timestamp} yok –> KeyError lar biter."""
        emoji = {
            "TRACE": "🔍",
            "DEBUG": "🐛",
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🔥",
        }.get(record["level"].name, "📝")

        module = record["name"].split(".")[-1]
        time_str = record["time"].strftime("%H:%M:%S")

        return (
            f"<green>{time_str}</green> | "
            f"{emoji} <level>{record['level'].name: <8}</level> | "
            f"<cyan>{module}</cyan>:<cyan>{record['function']}</cyan>:<cyan>{record['line']}</cyan> - "
            f"<level>{record['message']}</level>"
        )

    def _file_format(self, record):
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        )

    def _performance_format(self, record):
        extra = record["extra"]
        return (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message} | "
            f"duration={extra.get('duration','N/A')}ms | "
            f"memory={extra.get('memory','N/A')}MB"
        )

    # ──────────────────────────────── 3️⃣  Kısayol metotları ──────────────────────────
    @staticmethod
    def log_api_call(api, method, duration, success, **kw):
        logger.bind(api=api, method=method, duration=duration, success=success, **kw)\
              .info("API call")

    @staticmethod
    def log_interview_event(event_type, session_id, **kw):
        logger.bind(event_type=event_type, session_id=session_id, **kw)\
              .info("Interview event")

    @staticmethod
    def log_performance(op, duration, **kw):
        logger.bind(performance=True, duration=round(duration * 1000, 2), **kw)\
              .info(op)

    @staticmethod
    def log_error_with_context(err: Exception, ctx: dict):
        logger.bind(**ctx).exception(f"{type(err).__name__}: {err}")


# ──────────────────────────────── 4️⃣  Global kısayollar ─────────────────────────────
interview_logger = InterviewLogger()
log_api_call        = interview_logger.log_api_call
log_interview_event = interview_logger.log_interview_event
log_performance     = interview_logger.log_performance
log_error_with_context = interview_logger.log_error_with_context


# ──────────────────────────────── 5️⃣  Decorator’lar ────────────────────────────────
def log_execution_time(fn):
    """sync & async süre ölçer"""
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*a, **kw):
            t0 = time.time()
            try:
                return await fn(*a, **kw)
            finally:
                log_performance(fn.__name__, time.time() - t0)
        return wrapper
    else:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                log_performance(fn.__name__, time.time() - t0)
        return wrapper


def log_api_usage(api):
    def deco(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*a, **kw):
                t0 = time.time()
                try:
                    res = await fn(*a, **kw)
                    log_api_call(api, fn.__name__, time.time() - t0, True)
                    return res
                except Exception as e:
                    log_api_call(api, fn.__name__, time.time() - t0, False, error=str(e))
                    raise
            return wrapper
        else:
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                t0 = time.time()
                try:
                    res = fn(*a, **kw)
                    log_api_call(api, fn.__name__, time.time() - t0, True)
                    return res
                except Exception as e:
                    log_api_call(api, fn.__name__, time.time() - t0, False, error=str(e))
                    raise
            return wrapper
    return deco


# ──────────────────────────────── 6️⃣  Hızlı test ──────────────────────────────────
if __name__ == "__main__":
    logger.info("Info test")
    logger.success("Success test")
    log_performance("dummy_op", 0.123)
