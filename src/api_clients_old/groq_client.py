"""
Groq API İstemcisi - LLaMA 3.3 70B Model Erişimi

Bu modül, Groq API üzerinden LLaMA 3.3 70B modeline erişim sağlar.
Groq'un en büyük avantajı, çok hızlı inference (çıkarım) yapmasıdır.
Bu sayede mülakat sırasında beklemeler minimuma iner.

Tasarım Kararları:
1. Retry mekanizması: API çağrıları bazen başarısız olabilir
2. Streaming: Uzun yanıtları parça parça alarak deneyimi iyileştirir
3. Token sayımı: Maliyet kontrolü için önemli
"""

from typing import List, Dict, Optional, AsyncGenerator
import asyncio
from groq import Groq, AsyncGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config


class GroqClient:
    """
    Groq API ile etkileşimi yöneten istemci sınıfı.
    
    Bu sınıf, LLaMA 3.3 70B modeline güvenli ve verimli erişim sağlar.
    Hata yönetimi, retry mantığı ve performans optimizasyonlarını içerir.
    """
    
    def __init__(self):
        """Groq istemcisini başlat"""
        self.api_key = config.api.groq_api_key
        self.model = config.api.groq_model
        
        # Senkron ve asenkron istemciler
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        
        # İstatistikler
        self.total_tokens_used = 0
        self.total_requests = 0
        
        logger.info(f"Groq istemcisi başlatıldı. Model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        LLaMA 3.3 70B'den yanıt üret.
        
        Args:
            messages: Konuşma geçmişi [{"role": "user/assistant/system", "content": "..."}]
            temperature: Yaratıcılık seviyesi (0-2 arası, düşük = daha deterministik)
            max_tokens: Maksimum yanıt uzunluğu
            stream: Streaming kullanılsın mı?
            
        Returns:
            Model yanıtı
            
        Raises:
            Exception: API hatası durumunda
        """
        try:
            logger.debug(f"Groq'a istek gönderiliyor. Mesaj sayısı: {len(messages)}")
            
            # API çağrısı
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Streaming yanıt
                return self._handle_stream(response)
            else:
                # Normal yanıt
                content = response.choices[0].message.content
                
                # İstatistikleri güncelle
                self.total_requests += 1
                if hasattr(response, 'usage'):
                    self.total_tokens_used += response.usage.total_tokens
                    logger.debug(f"Token kullanımı: {response.usage.total_tokens}")
                
                logger.info("Groq yanıtı başarıyla alındı")
                return content
                
        except Exception as e:
            logger.error(f"Groq API hatası: {e}")
            raise
    
    async def _handle_stream(self, stream: AsyncGenerator) -> str:
        """
        Streaming yanıtı işle.
        
        Streaming, uzun yanıtlarda kullanıcı deneyimini iyileştirir.
        Yanıt parça parça gelir ve anında gösterilebilir.
        """
        full_response = ""
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                # Burada her chunk'ı UI'a gönderebiliriz
                
        return full_response
    
    def create_interview_prompt(
        self,
        job_info: Dict,
        candidate_info: Dict,
        phase: str,
        previous_qa: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """
        Mülakat için uygun prompt'u oluştur.
        
        Bu metod, mülakat bağlamına göre LLaMA'ya gönderilecek
        mesajları hazırlar. Her aşama için farklı yönergeler içerir.
        
        Args:
            job_info: İş pozisyonu bilgileri
            candidate_info: Aday bilgileri
            phase: Mevcut mülakat aşaması
            previous_qa: Önceki soru-cevaplar
            
        Returns:
            Mesaj listesi
        """
        # Sistem mesajı
        system_message = {
            "role": "system",
            "content": config.interview.system_prompt_template.format(
                position=job_info.get('title', 'Unknown Position'),
                company=job_info.get('company', 'Unknown Company'),
                candidate_name=candidate_info.get('name', 'Candidate'),
                job_requirements=self._format_requirements(job_info.get('requirements', {})),
                candidate_summary=self._format_candidate_summary(candidate_info)
            )
        }
        
        messages = [system_message]
        
        if previous_qa:
            for qa_item in previous_qa:
                role = qa_item.get("role")
                content = qa_item.get("text", "")
                if role == "ai":
                    messages.append({"role": "assistant", "content": content})
                elif role == "human":
                    messages.append({"role": "user", "content": content})
        
        # Aşamaya özel yönerge
        phase_instruction = self._get_phase_instruction(phase, len(previous_qa) if previous_qa else 0)
        messages.append({"role": "user", "content": phase_instruction})
        
        return messages
    
    def _format_requirements(self, requirements: Dict) -> str:
        """İş gereksinimlerini formatla"""
        formatted = []
        
        if 'technical_skills' in requirements:
            formatted.append(f"Technical Skills: {', '.join(requirements['technical_skills'])}")
        
        if 'experience_years' in requirements:
            formatted.append(f"Experience: {requirements['experience_years']} years")
        
        if 'education' in requirements:
            formatted.append(f"Education: {requirements['education']}")
        
        return '\n'.join(formatted)
    
    def _format_candidate_summary(self, candidate: Dict) -> str:
        """Aday özetini formatla"""
        summary = []
        
        if 'current_position' in candidate:
            summary.append(f"Current Position: {candidate['current_position']}")
        
        if 'years_experience' in candidate:
            summary.append(f"Total Experience: {candidate['years_experience']} years")
        
        if 'key_skills' in candidate:
            summary.append(f"Key Skills: {', '.join(candidate['key_skills'])}")
        
        return '\n'.join(summary)
    
    def _get_phase_instruction(self, phase: str, question_count: int) -> str:
        """
        Mülakat aşamasına göre yönerge döndür.
        
        Her aşamanın kendine özgü soru tipleri ve yaklaşımları var.
        Bu metod, LLaMA'ya hangi tür soru sorması gerektiğini söyler.
        """
        instructions = {
            "warmup": (
                "Start with a warm, welcoming question to make the candidate comfortable. "
                "Ask about their background or what interests them about this role. "
                "Keep it conversational and friendly."
            ),
            "technical": (
                "Ask a technical question related to the job requirements. "
                "Focus on their practical experience with the required technologies. "
                "Ask for specific examples from their past work."
            ),
            "behavioral": (
                "Ask a behavioral question using the STAR method. "
                "Focus on situations where they demonstrated key soft skills like "
                "teamwork, leadership, or problem-solving."
            ),
            "situational": (
                "Present a hypothetical scenario relevant to this role. "
                "Ask how they would handle the situation and why. "
                "The scenario should test their decision-making process."
            ),
            "closing": (
                "We're concluding the interview. Ask if they have any questions "
                "about the role, company, or next steps. Be prepared to provide "
                "general information about the hiring process."
            )
        }
        
        base_instruction = instructions.get(phase, "Ask a relevant follow-up question.")
        
        # Her zaman tek soru sormasını hatırlat
        return (
            f"{base_instruction}\n\n"
            f"IMPORTANT: Ask only ONE clear, specific question. "
            f"Do not ask multiple questions at once."
        )
    
    def get_statistics(self) -> Dict:
        """API kullanım istatistiklerini döndür"""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "average_tokens_per_request": (
                self.total_tokens_used / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "model": self.model
        }
    
    async def test_connection(self) -> bool:
        """
        API bağlantısını test et.
        
        Bu metod, sistemin başlangıcında API'nin çalıştığını
        doğrulamak için kullanılır.
        """
        try:
            logger.info("Groq API bağlantısı test ediliyor...")
            
            response = await self.generate_response(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Connection successful' in exactly 2 words."}
                ],
                max_tokens=10
            )
            
            if "successful" in response.lower():
                logger.info("✅ Groq API bağlantısı başarılı!")
                return True
            else:
                logger.warning(f"Beklenmeyen yanıt: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Groq API bağlantı hatası: {e}")
            return False


# Test için
if __name__ == "__main__":
    async def test():
        client = GroqClient()
        
        # Bağlantı testi
        if await client.test_connection():
            print("✅ Groq bağlantısı başarılı!")
            
            # Örnek soru üretimi
            messages = client.create_interview_prompt(
                job_info={"title": "Software Engineer", "company": "TechCorp"},
                candidate_info={"name": "Test User", "current_position": "Developer"},
                phase="warmup"
            )
            
            response = await client.generate_response(messages)
            print(f"\nÖrnek Soru: {response}")
            
            # İstatistikler
            print(f"\nİstatistikler: {client.get_statistics()}")
        else:
            print("❌ Groq bağlantısı başarısız!")
    
    asyncio.run(test())