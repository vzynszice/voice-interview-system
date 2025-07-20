import argostranslate.package
import argostranslate.translate
from loguru import logger
class ArgosTranslatorClient:
    """
    Argos Translate kullanarak tamamen lokal çeviri yapar.
    """
    def __init__(self, from_code="en", to_code="tr"):
        self.from_code = from_code
        self.to_code = to_code
        
        # Gerekli dil paketlerini kontrol et ve indir
        self._check_and_install_packages()
        
        self.from_lang = argostranslate.translate.get_language_from_code(self.from_code)
        self.to_lang = argostranslate.translate.get_language_from_code(self.to_code)
        self.translation = self.from_lang.get_translation(self.to_lang)

        if not self.translation:
            raise RuntimeError(f"Argos Translate için {from_code} -> {to_code} çeviri paketi bulunamadı!")
            
        logger.success(f"Argos Translator başlatıldı: {from_code} -> {to_code}")

    def _check_and_install_packages(self):
        """Dil paketlerinin yüklü olup olmadığını kontrol eder, değilse indirir."""
        available_packages = argostranslate.package.get_installed_packages()
        available_codes = [pkg.from_code for pkg in available_packages] + [pkg.to_code for pkg in available_packages]
        
        if self.from_code not in available_codes or self.to_code not in available_codes:
            logger.info("Gerekli Argos Translate dil paketleri indiriliyor...")
            argostranslate.package.update_package_index()
            available_packages_to_install = argostranslate.package.get_available_packages()
            
            for package in available_packages_to_install:
                if package.from_code == self.from_code and package.to_code == self.to_code:
                    logger.info(f"Paket indiriliyor: {package}")
                    package.install()
                    return

    def translate(self, text: str) -> str:
        """Verilen metni çevirir."""
        if not self.translation:
            logger.error("Çeviri motoru başlatılamadı, orijinal metin döndürülüyor.")
            return text
        
        translated_text = self.translation.translate(text)
        logger.debug(f"Çeviri: '{text}' -> '{translated_text}'")
        return translated_text
