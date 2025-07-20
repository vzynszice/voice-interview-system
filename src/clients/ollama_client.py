"""
Ollama API İstemcisi - Lokal LLM Erişimi (Gemma3:1B)

Bu modül, Ollama üzerinden Gemma3:1B modeline erişim sağlar.
Groq API'nin yerini alacak şekilde tasarlanmıştır.
"""

import asyncio
import json
from typing import List, Dict, Optional, AsyncGenerator
import aiohttp
from loguru import logger

from src.utils.error_handlers import api_retry_handler
from config import config


class OllamaClient:
    """
    Ollama API ile lokal LLM etkileşimini yöneten istemci.
    
    Gemma3:1B modelini kullanarak hızlı ve gizli çıkarım sağlar.
    """
    
    def __init__(self):
        """Ollama istemcisini başlat"""
        # Ollama varsayılan olarak localhost:11434'te çalışır
        self.base_url = "http://localhost:11434"
        self.model = "gemma3:1b"  # Kullanıcının istediği model
        
        # Async HTTP client
        self.session = None
        
        # İstatistikler
        self.total_tokens_used = 0
        self.total_requests = 0
        
        logger.info(f"Ollama istemcisi başlatıldı. Model: {self.model}")
    
    async def _ensure_session(self):
        """HTTP session'ı garanti et"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    @api_retry_handler()
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Gemma3:1B'den yanıt üret.
        
        Args:
            messages: Konuşma geçmişi [{"role": "user/assistant/system", "content": "..."}]
            temperature: Yaratıcılık seviyesi (0-2 arası)
            max_tokens: Maksimum yanıt uzunluğu
            stream: Streaming kullanılsın mı?
            
        Returns:
            Model yanıtı
        """
        try:
            logger.debug(f"Ollama'ya istek gönderiliyor. Mesaj sayısı: {len(messages)}")
            
            await self._ensure_session()
            
            # Ollama API formatına dönüştür
            prompt = self._format_messages(messages)
            
            if stream:
                # Streaming yanıt
                return await self._generate_streaming(prompt, temperature, max_tokens)
            else:
                # Normal yanıt
                url = f"{self.base_url}/api/generate"
                
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "options": {
                        "num_predict": max_tokens,
                        "stop": ["Human:", "User:", "Assistant:"]
                    },
                    "stream": False
                }
                
                async with self.session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API hatası: {response.status} - {error_text}")
                    
                    result = await response.json()
                    content = result.get("response", "")
                    
                    # İstatistikleri güncelle
                    self.total_requests += 1
                    if "total_duration" in result:
                        logger.debug(f"Yanıt süresi: {result['total_duration'] / 1e9:.2f}s")
                    
                    logger.info("Ollama yanıtı başarıyla alındı")
                    return content.strip()
                    
        except Exception as e:
            logger.error(f"Ollama API hatası: {e}")
            raise
    
    async def _generate_streaming(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Streaming yanıt üret"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens
            },
            "stream": True
        }
        
        async with self.session.post(url, json=payload) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Mesaj listesini Ollama'nın anlayacağı formata dönüştür.
        
        Gemma modelleri için uygun format kullanılır.
        """
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Sistem mesajını en başa koy
                formatted.insert(0, f"System: {content}\n")
            elif role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        # Son olarak Assistant: ekle ki model cevap vermeye başlasın
        formatted.append("Assistant:")
        
        return "\n\n".join(formatted)
    
    def create_interview_prompt(
        self,
        job_info: Dict,
        candidate_info: Dict,
        phase: str,
        previous_qa: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """
        Mülakat için uygun prompt'u oluştur.
        
        Bu metod Groq client'tan kopyalandı, aynı interface'i koruyoruz.
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
        
        # Önceki konuşmaları ekle
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
        """Mülakat aşamasına göre yönerge döndür"""
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
        
        return (
            f"{base_instruction}\n\n"
            f"IMPORTANT: Ask only ONE clear, specific question. "
            f"Do not ask multiple questions at once."
        )
    
    def get_statistics(self) -> Dict:
        """API kullanım istatistiklerini döndür"""
        return {
            "total_requests": self.total_requests,
            "model": self.model,
            "base_url": self.base_url
        }
    
    # ollama_client.py -> test_connection metodunun OLMASI GEREKEN DOĞRU HALİ
    async def test_connection(self) -> bool:
        """Ollama bağlantısını ve modelin cevap verip vermediğini test et."""
        try:
            logger.info("Ollama API bağlantısı test ediliyor...")
            await self._ensure_session()

            # 1. Ollama servisinin ayakta olup olmadığını kontrol et
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    logger.error("Ollama servisi çalışmıyor veya ulaşılamıyor!")
                    return False
                
                data = await response.json()
                models = [m["name"] for m in data.get("models", [])]
                
                # 2. İhtiyaç duyulan modelin yüklü olup olmadığını kontrol et
                if self.model not in models:
                    logger.warning(f"'{self.model}' modeli bulunamadı! Mevcut modeller: {models}")
                    logger.info(f"Modeli indirmek için: ollama pull {self.model}")
                    return False
            
            # 3. Modele çok basit bir soru sorup cevap alıp alamadığımızı kontrol et
            response = await self.generate_response(
                messages=[{"role": "user", "content": "1+1 kaç eder?"}],
                temperature=0,
                max_tokens=5
            )
            
            # 4. Cevabın boş olup olmadığını kontrol et. Boş değilse, bağlantı başarılıdır.
            if response and response.strip():
                logger.info("✅ Ollama API bağlantısı başarılı!")
                return True
            else:
                logger.warning(f"Ollama'dan boş veya anlamsız yanıt alındı: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ollama API bağlantı hatası: {e}")
            logger.info("Ollama'nın çalıştığından emin olun: `ollama serve`")
            return False
    
    async def close(self):
        """Session'ı kapat"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'session') and self.session:
            try:
                asyncio.get_event_loop().run_until_complete(self.close())
            except:
                pass  # Event loop yoksa sessizce geç


# Test için
if __name__ == "__main__":
    async def test():
        client = OllamaClient()
        
        try:
            # Bağlantı testi
            if await client.test_connection():
                print("✅ Ollama bağlantısı başarılı!")
                
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
                print("❌ Ollama bağlantısı başarısız!")
        finally:
            await client.close()
    
    asyncio.run(test())