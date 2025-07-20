"""
Google Cloud Translation v3 API İstemcisi – Yüksek Kaliteli Çeviri

DeepL yerine Google Cloud'un gelişmiş Translation hizmetine geçildi.
Bu modül, Türkçe–İngilizce (ve tersi) çevirileri Google Cloud üzerinde
hızlı ve bağlam duyarlı biçimde gerçekleştirir.

Avantajlar
----------
1. 130+ dil desteği, bölge seçimi (düşük gecikme)
2. Terim sözlükleri (glossary) ve formality seçenekleri
3. Otomatik kaynak dil algılama
4. Streaming gerektirmez; yanıtlar tek istekte gelir
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from google.cloud import translate_v3 as translate  # type: ignore
from google.api_core.exceptions import GoogleAPIError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config

# ---------------------------------------------------------------------------
# İstemci Sınıfı
# ---------------------------------------------------------------------------


class GCTClient:
    """Google Cloud Translation kullanarak Türkçe–İngilizce çeviri yapar."""

    def __init__(self) -> None:
        # Kimlik bilgileri GOOGLE_APPLICATION_CREDENTIALS üzerinden yüklenir
        self.project_id: str | None = config.api.google_cloud_project_id
        if not self.project_id:
            # Credentials içinden proje id’sini al (service‑account içindeki project_id)
            creds_path: Path = config.api.google_application_credentials
            import json, os

            with open(creds_path, "r", encoding="utf-8") as f:
                self.project_id = json.load(f).get("project_id")

        if not self.project_id:
            raise RuntimeError(
                "Google Cloud project_id bulunamadı. .env'de GOOGLE_CLOUD_PROJECT_ID ekleyin."
            )

        # İstemci
        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{self.project_id}/locations/{config.api.gct_location}"

        # Varsayılan diller
        self.source_lang_default = config.api.gct_source_lang  # "tr"
        self.target_lang_default = config.api.gct_target_lang  # "en"

        # Önbellek
        self._cache: dict[str, str] = {}
        self._cache_max = 1000
        self.cache_hits = 0

        # İstatistikler
        self.total_translations = 0
        self.total_chars = 0

        logger.info(
            f"Google Translation istemcisi başlatıldı. Bölge: {config.api.gct_location}"
        )

    # ---------------------------------------------------------------------
    # Yardımcı Metotlar
    # ---------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(text: str, src: str | None, tgt: str) -> str:
        key = f"{text}|{src or 'auto'}|{tgt}"
        return hashlib.md5(key.encode()).hexdigest()

    def _add_cache(self, key: str, value: str) -> None:
        if len(self._cache) >= self._cache_max:
            # En eski %20’yi sil
            for _ in range(int(self._cache_max * 0.2)):
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    # ---------------------------------------------------------------------
    # Ana Çeviri Metodu
    # ---------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def translate(
        self,
        text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
        glossary_name: str | None = None,
        model: str | None = None,
    ) -> str:
        """Metni çevirir – hata durumunda otomatik yeniden dener."""

        if not text.strip():
            return ""

        src = source_lang or self.source_lang_default
        tgt = target_lang or self.target_lang_default

        cache_key = self._make_cache_key(text, src, tgt)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        logger.debug(f"Çeviri isteği: {src or 'auto'} → {tgt}, uzunluk={len(text)}")

        def _blocking_call() -> str:
            req = translate.TranslateTextRequest(
                parent=self.parent,
                contents=[text],
                mime_type="text/plain",
                target_language_code=tgt,
                source_language_code=src if src != "auto" else None,
                glossary_config=translate.TranslateTextGlossaryConfig(
                    glossary=f"{self.parent}/glossaries/{glossary_name}"
                )
                if glossary_name
                else None,
                model=model,  # Bölgesel özel modeli belirtmek isterseniz
            )
            resp = self.client.translate_text(request=req)
            return resp.translations[0].translated_text

        try:
            translated_text: str = await asyncio.get_event_loop().run_in_executor(
                None, _blocking_call
            )
        except GoogleAPIError as exc:
            logger.error(f"Google Translation API hatası: {exc}")
            raise

        self.total_translations += 1
        self.total_chars += len(text)
        self._add_cache(cache_key, translated_text)

        return translated_text

    # ------------------------------------------------------------------
    # Konuşma Geçmişini Çevir
    # ------------------------------------------------------------------

    async def translate_conversation(
        self, messages: List[Dict[str, str]], direction: str = "tr_to_en"
    ) -> List[Dict[str, str]]:
        src = "tr" if direction == "tr_to_en" else "en"
        tgt = "en" if direction == "tr_to_en" else "tr"

        translated: list[Dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                translated.append(msg)
                continue
            translated_content = await self.translate(msg.get("content", ""), src, tgt)
            translated.append({"role": msg.get("role"), "content": translated_content})
        return translated

    # ------------------------------------------------------------------
    # İstatistik Metotları
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, str | int | float]:
        hit_rate = (
            self.cache_hits / (self.total_translations + self.cache_hits) * 100
            if (self.total_translations + self.cache_hits) > 0
            else 0
        )
        return {
            "total_translations": self.total_translations,
            "total_characters": self.total_chars,
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{hit_rate:.1f}%",
        }

    # ------------------------------------------------------------------
    # Sağlık Testi
    # ------------------------------------------------------------------

    async def test_connection(self) -> bool:
        try:
            logger.info("Google Translation API bağlantısı test ediliyor…")
            result = await self.translate("Merhaba dünya", "tr", "en")
            if result.lower().startswith("hello"):
                logger.info("✅ Google Translation API bağlantısı başarılı!")
                return True
            logger.warning(f"Beklenmeyen sonuç: {result}")
            return False
        except Exception as e:
            logger.error(f"❌ Google Translation API bağlantı hatası: {e}")
            return False


# ---------------------------------------------------------------------------
# Modül Doğrudan Çalıştırıldığında Mini Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _quick_test():
        client = GCTClient()
        ok = await client.test_connection()
        if ok:
            tr = await client.translate("Python geliştirme deneyimim var.")
            print("Çeviri:", tr)
            print("İstatistikler:", client.get_statistics())

    asyncio.run(_quick_test())
