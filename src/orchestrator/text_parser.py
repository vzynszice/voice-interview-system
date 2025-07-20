"""
Text Parser - Metni Akıllıca Cümlelere Bölen Modül

Bu modül, gelen metni cümlelere ayırır. Basit nokta araması yapmak yerine,
kısaltmalar, sayılar ve özel durumları dikkate alarak akıllı bir ayrım yapar.
"""

import re
from typing import List, Generator
from loguru import logger


class SentenceParser:
    """
    Metni cümlelere ayıran akıllı parser.
    
    Bu sınıf, Türkçe ve İngilizce metinleri doğru şekilde cümlelere
    ayırabilmek için gelişmiş kurallar kullanır.
    """
    
    def __init__(self):
        # Cümle sonu olarak kabul EDİLMEYECEK kısaltmalar
        self.abbreviations = {
            # Türkçe kısaltmalar
            'Dr', 'Prof', 'Doç', 'Yrd', 'Uzm', 'Av', 'Müh', 'Öğr', 'Gör',
            'vb', 'vs', 'vd', 'yy', 'sy', 'sf', 'cm', 'km', 'gr', 'kg',
            'TL', 'USD', 'EUR', 'Ltd', 'Şti', 'AŞ', 'AS', 'no', 'No',
            # İngilizce kısaltmalar
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'Ph.D', 'M.D',
            'B.A', 'M.A', 'B.S', 'M.S', 'LLC', 'Inc', 'Corp', 'Co',
            'etc', 'vs', 'ie', 'eg', 'cf', 'al', 'St', 'Ave', 'Blvd'
        }
        
        # Cümle sonu işaretleri
        self.sentence_endings = '.!?'
        
        # Özel pattern'ler
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Regex pattern'lerini önceden derle (performans için)"""
        # Sayı formatları (1.000, 3.14, vs.)
        self.number_pattern = re.compile(r'\d+\.\d+')
        
        # Web adresleri ve e-mail
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        
        # Üç nokta
        self.ellipsis_pattern = re.compile(r'\.{3,}')
    
    def parse_sentences(self, text: str) -> List[str]:
        """
        Metni cümlelere ayır.
        
        Args:
            text: Cümlelere ayrılacak metin
            
        Returns:
            Cümle listesi
        """
        if not text or not text.strip():
            return []
        
        # Metni temizle ama orijinal boşlukları koru
        text = text.strip()
        
        # Özel durumları geçici olarak değiştir
        text = self._protect_special_cases(text)
        
        # Cümlelere ayır
        sentences = self._split_into_sentences(text)
        
        # Geçici değişiklikleri geri al
        sentences = [self._restore_special_cases(s) for s in sentences]
        
        # Boş cümleleri filtrele
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.debug(f"Metin {len(sentences)} cümleye ayrıldı")
        return sentences
    
    def parse_streaming(self, text: str, buffer: str = "") -> Generator[str, None, str]:
        """
        Streaming metin için cümle parser.
        
        Bu metod, henüz tamamlanmamış metinlerle çalışabilir.
        LLM'den gelen streaming response'lar için idealdir.
        
        Args:
            text: Yeni gelen metin parçası
            buffer: Önceki tamamlanmamış metin
            
        Yields:
            Tamamlanmış cümleler
            
        Returns:
            Yeni buffer (tamamlanmamış kısım)
        """
        # Buffer'a yeni metni ekle
        combined = buffer + text
        
        # Cümle sonu işareti var mı kontrol et
        last_ending = -1
        for i, char in enumerate(combined):
            if char in self.sentence_endings:
                # Bunun gerçek bir cümle sonu mu kontrol et
                if self._is_sentence_end(combined, i):
                    # Cümleyi yield et
                    yield combined[last_ending + 1:i + 1].strip()
                    last_ending = i
        
        # Kalan kısmı buffer olarak döndür
        return combined[last_ending + 1:]
    
    def _protect_special_cases(self, text: str) -> str:
        """
        Özel durumları geçici placeholder'larla değiştir.
        
        Bu sayede kısaltmalar, URL'ler vs. yanlışlıkla bölünmez.
        """
        # Üç noktayı koru
        text = self.ellipsis_pattern.sub('___ELLIPSIS___', text)
        
        # URL'leri koru
        text = self.url_pattern.sub(lambda m: m.group().replace('.', '___DOT___'), text)
        
        # Email'leri koru
        text = self.email_pattern.sub(lambda m: m.group().replace('.', '___DOT___'), text)
        
        # Sayıları koru (3.14, 1.000 gibi)
        text = self.number_pattern.sub(lambda m: m.group().replace('.', '___DOT___'), text)
        
        # Kısaltmaları koru
        for abbr in self.abbreviations:
            # Hem "Dr." hem "Dr " formatlarını yakala
            text = re.sub(
                rf'\b{re.escape(abbr)}\.(?=\s|$)', 
                f'{abbr}___DOT___', 
                text
            )
        
        return text
    
    def _restore_special_cases(self, text: str) -> str:
        """Placeholder'ları geri çevir"""
        text = text.replace('___DOT___', '.')
        text = text.replace('___ELLIPSIS___', '...')
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Metni cümle sonlarından böl"""
        sentences = []
        current = []
        
        i = 0
        while i < len(text):
            current.append(text[i])
            
            # Cümle sonu mu kontrol et
            if text[i] in self.sentence_endings:
                # Sonraki karaktere bak
                if i + 1 < len(text):
                    next_char = text[i + 1]
                    # Eğer sonraki karakter boşluk veya yeni satırsa, cümle bitmiştir
                    if next_char in ' \n\t':
                        # Ama önce tırnak kontrolü yap
                        if i + 2 < len(text) and text[i + 1] == ' ' and text[i + 2] in '"\'""':
                            # Tırnağı da cümleye dahil et
                            current.append(text[i + 1])
                            current.append(text[i + 2])
                            i += 2
                        sentences.append(''.join(current).strip())
                        current = []
                    # Eğer sonraki karakter küçük harfse, muhtemelen kısaltma
                    elif next_char.islower():
                        pass  # Cümle devam ediyor
                else:
                    # Metnin sonu
                    sentences.append(''.join(current).strip())
                    current = []
            
            i += 1
        
        # Kalan varsa ekle
        if current:
            sentences.append(''.join(current).strip())
        
        return sentences
    
    def _is_sentence_end(self, text: str, pos: int) -> bool:
        """
        Verilen pozisyondaki işaretin gerçek cümle sonu olup olmadığını kontrol et.
        
        Bu metod, bağlamsal analiz yaparak false positive'leri önler.
        """
        if pos >= len(text) - 1:
            return True  # Metnin sonu
        
        # Sonraki karakter kontrolü
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            
            # Boşluk veya yeni satır varsa muhtemelen cümle sonu
            if next_char in ' \n\t':
                # Ama öncesinde kısaltma var mı?
                word_before = self._get_word_before(text, pos)
                if word_before in self.abbreviations:
                    # Kısaltmadan sonra büyük harf geliyorsa yine de cümle sonu olabilir
                    if pos + 2 < len(text) and text[pos + 2].isupper():
                        return True
                    return False
                return True
            
            # Tırnak işareti kontrolü
            elif next_char in '"\'""':
                return True
            
            # Büyük harf kontrolü (yeni cümle başlangıcı)
            elif next_char.isupper():
                return True
        
        return False
    
    def _get_word_before(self, text: str, pos: int) -> str:
        """Verilen pozisyondan önceki kelimeyi al"""
        # Geriye doğru git ve kelime başlangıcını bul
        start = pos
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        
        return text[start:pos]


# Test fonksiyonu
if __name__ == "__main__":
    parser = SentenceParser()
    
    # Test metinleri
    test_texts = [
        "Merhaba Dr. Ahmet. Bugün nasılsınız? Umarım iyisinizdir.",
        "Fiyat 3.500 TL'dir. Ödeme için son tarih 15.03.2024.",
        "Web sitemiz www.example.com adresindedir. Detaylar için bakınız.",
        "Hmm... Düşünmem gerekiyor. Belki yarın cevap veririm.",
        'O bana "Yarın gel." dedi. Ben de "Tamam." dedim.',
    ]
    
    for text in test_texts:
        print(f"\nMetin: {text}")
        sentences = parser.parse_sentences(text)
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")