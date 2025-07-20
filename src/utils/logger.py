# src/utils/logger.py (Tam ve Doğru Hali)

import sys
import asyncio
import functools
import time
from pathlib import Path
from loguru import logger

# config'i doğrudan import etmek yerine, bu fonksiyon config nesnesini parametre olarak alabilir.
# Ama şimdilik basit tutmak için, bu dosyanın config'i import ettiğini varsayalım.
# Daha modüler bir yapı için, setup_logger(config) şeklinde de tasarlanabilir.

def setup_logging(log_level: str = "INFO", base_dir: Path = Path(".")):
    """
    Loguru için merkezi ve gelişmiş loglama kurulumu yapar.
    """
    logger.remove()  # eski handler’ları temizle

    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Konsol formatlayıcısı
    def console_format(record):
        emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(record["level"].name, "📝")
        time_str = record["time"].strftime("%H:%M:%S")
        return (
            f"<green>{time_str}</green> | {emoji} <level>{record['level'].name: <8}</level> | "
            f"<cyan>{record['name'].split('.')[-1]}</cyan>:<cyan>{record['function']}</cyan> - "
            f"<level>{record['message']}</level>\n"
        )

    # Konsol (renkli ve sade)
    logger.add(sys.stderr, format=console_format, level=log_level.upper(), colorize=True)

    # Ana log dosyası (detaylı)
    logger.add(
        log_dir / "interview_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # Hata log dosyası (sadece hatalar için)
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        rotation="1 week",
        backtrace=True, # Hatalar için tam traceback'i kaydet
        diagnose=True,
    )

    logger.success("Merkezi loglama sistemi başarıyla yapılandırıldı.")