"""
OpenAI ChatGPT API İstemcisi - Sentetik Veri Üretimi

Bu modül, ChatGPT'yi kullanarak gerçekçi iş ilanları ve şirket bilgileri üretir.
Neden sentetik veri? Çünkü her test için gerçek iş ilanı bulmak yerine,
ihtiyacımıza özel, tutarlı ve gerçekçi veriler üretebiliriz.

Kullanım Alanları:
1. İş ilanı (Job Post) üretimi
2. Şirket profili oluşturma
3. Pozisyon gereksinimleri belirleme
4. Mülakat soruları önerisi
"""

import json
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import random

from config import config


class ChatGPTClient:
    """
    OpenAI ChatGPT ile sentetik veri üreten istemci.
    
    Bu sınıf, mülakat sistemi için gerekli olan gerçekçi verileri üretir.
    Üretilen veriler, gerçek iş ilanlarından ayırt edilemeyecek kalitededir.
    """
    
    def __init__(self):
        """ChatGPT istemcisini başlat"""
        self.api_key = config.api.openai_api_key
        
        # Async OpenAI istemcisi
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Model ayarları
        self.model = "gpt-4o-mini"  # En güncel ve yetenekli model
        self.temperature = 0.8  # Yaratıcılık için biraz yüksek
        
        # İstatistikler
        self.total_generations = 0
        self.total_tokens_used = 0
        
        logger.info(f"ChatGPT istemcisi başlatıldı. Model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_job_post(
        self,
        position_title: str,
        company_type: Optional[str] = None,
        seniority_level: str = "mid",
        location: str = "Istanbul, Turkey",
        industry: Optional[str] = None
    ) -> Dict:
        """
        Gerçekçi bir iş ilanı üret.
        
        Args:
            position_title: Pozisyon başlığı (örn: "Full Stack Developer")
            company_type: Şirket tipi (startup, enterprise, vs.)
            seniority_level: Kıdem seviyesi (junior, mid, senior, lead)
            location: Konum
            industry: Sektör
            
        Returns:
            İş ilanı verisi (JSON formatında)
        """
        try:
            logger.info(f"İş ilanı üretiliyor: {position_title}")
            
            # Prompt'u hazırla
            prompt = self._create_job_post_prompt(
                position_title,
                company_type,
                seniority_level,
                location,
                industry
            )
            
            # ChatGPT'ye gönder
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR professional who creates detailed and realistic job postings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}  # JSON yanıt garantisi
            )
            
            # Yanıtı parse et
            content = response.choices[0].message.content
            job_post = json.loads(content)
            
            # İstatistikleri güncelle
            self.total_generations += 1
            if hasattr(response, 'usage'):
                self.total_tokens_used += response.usage.total_tokens
            
            # ID ve metadata ekle
            job_post['jobId'] = self._generate_job_id()
            job_post['metadata'] = {
                'createdDate': datetime.now().isoformat(),
                'generatedBy': 'ChatGPT',
                'status': 'active'
            }
            
            logger.info(f"İş ilanı başarıyla üretildi: {job_post['jobId']}")
            
            return job_post
            
        except Exception as e:
            logger.error(f"İş ilanı üretme hatası: {e}")
            raise
    
    def _create_job_post_prompt(
        self,
        position_title: str,
        company_type: Optional[str],
        seniority_level: str,
        location: str,
        industry: Optional[str]
    ) -> str:
        """
        İş ilanı üretimi için detaylı prompt oluştur.
        
        Prompt kalitesi, üretilen verinin kalitesini doğrudan etkiler.
        Bu yüzden çok detaylı ve yapılandırılmış bir prompt kullanıyoruz.
        """
        # Şirket tipi belirtilmemişse rastgele seç
        if not company_type:
            company_type = random.choice([
                "fast-growing startup",
                "established tech company",
                "multinational corporation",
                "innovative scale-up",
                "digital agency"
            ])
        
        # Sektör belirtilmemişse pozisyona göre tahmin et
        if not industry:
            if "developer" in position_title.lower() or "engineer" in position_title.lower():
                industry = "Technology"
            elif "designer" in position_title.lower():
                industry = "Design & Creative"
            elif "manager" in position_title.lower():
                industry = "Management"
            else:
                industry = "General Business"
        
        prompt = f"""Create a detailed and realistic job posting for a {seniority_level}-level {position_title} position.

Company Context:
- Type: {company_type}
- Location: {location}
- Industry: {industry}

Generate a comprehensive job posting in JSON format with the following structure:

{{
    "basicInfo": {{
        "title": "exact position title",
        "department": "relevant department",
        "location": "city, country",
        "employmentType": "full-time/part-time/contract",
        "experienceLevel": "junior/mid/senior/lead",
        "salaryRange": {{
            "min": number,
            "max": number,
            "currency": "TRY/USD/EUR"
        }}
    }},
    "company": {{
        "name": "realistic company name",
        "description": "2-3 sentences about the company",
        "size": "1-10/11-50/51-200/201-500/500+",
        "culture": ["list of", "culture values"],
        "benefits": ["list of", "company benefits"]
    }},
    "requirements": {{
        "education": {{
            "minimumDegree": "high-school/bachelor/master/phd",
            "preferredFields": ["relevant fields"],
            "mandatory": boolean
        }},
        "experience": {{
            "minimumYears": number,
            "requiredExperience": ["list of required experiences"],
            "preferredExperience": ["list of nice-to-have experiences"]
        }},
        "skills": {{
            "technical": [
                {{
                    "skillName": "skill name",
                    "level": "beginner/intermediate/advanced/expert",
                    "yearsRequired": number,
                    "mandatory": boolean
                }}
            ],
            "soft": ["communication", "teamwork", "etc."],
            "languages": [
                {{
                    "language": "English/Turkish/etc",
                    "proficiency": "basic/conversational/professional/native"
                }}
            ]
        }}
    }},
    "responsibilities": [
        "list of",
        "key responsibilities",
        "and duties"
    ],
    "preferredQualifications": [
        "nice to have",
        "qualifications"
    ],
    "interviewProcess": {{
        "stages": ["Phone Screen", "Technical Interview", "Team Fit", "Final Round"],
        "estimatedDuration": "2-3 weeks",
        "notes": "Any special notes about the process"
    }}
}}

Make it realistic and detailed. Use industry-standard terminology and requirements.
For a {location} based position, consider local market conditions and expectations."""
        
        return prompt
    
    async def generate_company_profile(
        self,
        company_name: Optional[str] = None,
        industry: str = "Technology",
        size: str = "51-200"
    ) -> Dict:
        """
        Gerçekçi bir şirket profili üret.
        
        Args:
            company_name: Şirket adı (opsiyonel)
            industry: Sektör
            size: Şirket büyüklüğü
            
        Returns:
            Şirket profil verisi
        """
        prompt = f"""Create a realistic company profile for a {size} employee company in the {industry} industry.
{f'Company name: {company_name}' if company_name else 'Generate a realistic company name.'}

Return a JSON object with:
{{
    "name": "company name",
    "industry": "{industry}",
    "size": "{size}",
    "founded": year,
    "headquarters": "city, country",
    "description": "detailed company description",
    "mission": "company mission statement",
    "values": ["core", "company", "values"],
    "products_services": ["main products or services"],
    "tech_stack": ["if applicable"],
    "culture": {{
        "work_style": "remote/hybrid/office",
        "team_structure": "description",
        "growth_opportunities": "description"
    }},
    "benefits": [
        "competitive salary",
        "health insurance",
        "other benefits"
    ],
    "website": "https://example.com",
    "social_media": {{
        "linkedin": "url",
        "twitter": "url"
    }}
}}"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a business analyst creating realistic company profiles."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def enhance_job_requirements(
        self,
        position_title: str,
        basic_requirements: List[str]
    ) -> Dict:
        """
        Temel gereksinimleri detaylandır ve zenginleştir.
        
        Basit bir gereksinim listesini, detaylı ve yapılandırılmış
        bir gereksinim setine dönüştürür.
        
        Args:
            position_title: Pozisyon başlığı
            basic_requirements: Temel gereksinimler listesi
            
        Returns:
            Detaylı gereksinimler
        """
        prompt = f"""Given the position "{position_title}" and these basic requirements:
{json.dumps(basic_requirements, indent=2)}

Create a comprehensive and detailed requirements structure:

{{
    "technical_skills": [
        {{
            "category": "Programming Languages/Frameworks/Tools/etc",
            "skills": [
                {{
                    "name": "skill name",
                    "level": "required level",
                    "years": "years of experience",
                    "details": "specific details about usage"
                }}
            ]
        }}
    ],
    "soft_skills": [
        {{
            "skill": "skill name",
            "importance": "critical/high/medium",
            "description": "why this skill matters for this role"
        }}
    ],
    "experience_areas": [
        {{
            "area": "area of experience",
            "years": "required years",
            "details": ["specific experiences expected"]
        }}
    ],
    "certifications": [
        {{
            "name": "certification name",
            "required": boolean,
            "alternatives": ["alternative certs"]
        }}
    ],
    "domain_knowledge": [
        "specific industry or domain knowledge required"
    ]
}}"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert recruiter who creates detailed job requirements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Biraz daha düşük, tutarlılık için
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def generate_interview_questions(
        self,
        job_post: Dict,
        question_type: str = "mixed",
        count: int = 10
    ) -> List[Dict]:
        """
        Pozisyona özel mülakat soruları üret.
        
        Bu fonksiyon, iş ilanına göre özelleştirilmiş sorular üretir.
        Sorular, gerçek mülakatlardan ayırt edilemez kalitededir.
        
        Args:
            job_post: İş ilanı verisi
            question_type: Soru tipi (technical/behavioral/situational/mixed)
            count: Üretilecek soru sayısı
            
        Returns:
            Soru listesi
        """
        position = job_post.get('basicInfo', {}).get('title', 'Unknown Position')
        requirements = job_post.get('requirements', {})
        
        prompt = f"""Generate {count} interview questions for a {position} position.

Job Requirements Summary:
{json.dumps(requirements, indent=2)}

Question Type: {question_type}

Generate questions in this JSON format:
{{
    "questions": [
        {{
            "id": "Q1",
            "type": "technical/behavioral/situational",
            "question": "the actual question",
            "intent": "what this question aims to assess",
            "follow_ups": ["potential follow-up questions"],
            "good_answer_hints": ["what makes a good answer"],
            "red_flags": ["what to watch out for"]
        }}
    ]
}}

Make questions specific to the role and requirements. Include a mix of:
- Role-specific technical questions
- Problem-solving scenarios
- Past experience questions
- Culture fit questions"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an experienced interviewer creating thoughtful interview questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('questions', [])
    
    def _generate_job_id(self) -> str:
        """
        Benzersiz bir iş ilanı ID'si üret.
        
        ID formatı: JOB_YYYYMMDD_XXXX (örn: JOB_20240115_A7B3)
        """
        date_part = datetime.now().strftime("%Y%m%d")
        random_part = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=4))
        return f"JOB_{date_part}_{random_part}"
    
    async def create_complete_interview_package(
        self,
        position_title: str,
        candidate_level: str = "mid",
        interview_duration: int = 45
    ) -> Dict:
        """
        Tam bir mülakat paketi oluştur.
        
        Bu fonksiyon, bir pozisyon için gereken tüm verileri üretir:
        - Şirket profili
        - İş ilanı
        - Mülakat soruları
        - Değerlendirme kriterleri
        
        Args:
            position_title: Pozisyon başlığı
            candidate_level: Aday seviyesi
            interview_duration: Mülakat süresi (dakika)
            
        Returns:
            Komple mülakat paketi
        """
        logger.info(f"Komple mülakat paketi oluşturuluyor: {position_title}")
        
        # 1. Önce şirket profili oluştur
        company = await self.generate_company_profile(
            industry="Technology",
            size="51-200"
        )
        
        # 2. İş ilanı oluştur
        job_post = await self.generate_job_post(
            position_title=position_title,
            company_type="tech company",
            seniority_level=candidate_level,
            location="Istanbul, Turkey"
        )
        
        # Şirket bilgilerini iş ilanına ekle
        job_post['company'] = company
        
        # 3. Mülakat soruları oluştur
        questions = await self.generate_interview_questions(
            job_post=job_post,
            question_type="mixed",
            count=14  # Mülakat ayarlarına göre
        )
        
        # 4. Mülakat ayarlarını ekle
        job_post['interviewSettings'] = {
            'estimatedDuration': f"{interview_duration} minutes",
            'questionDistribution': {
                'warmup': 2,
                'technical': 4,
                'behavioral': 3,
                'situational': 3,
                'closing': 2
            },
            'generatedQuestions': questions,
            'evaluationCriteria': {
                'weights': {
                    'technicalCompetency': 0.35,
                    'analyticalThinking': 0.20,
                    'problemSolving': 0.20,
                    'focusSkills': 0.15,
                    'communicationSkills': 0.10
                }
            }
        }
        
        # 5. Paketi oluştur
        package = {
            'packageId': f"PKG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'createdAt': datetime.now().isoformat(),
            'position': position_title,
            'level': candidate_level,
            'company': company,
            'jobPost': job_post,
            'interviewQuestions': questions,
            'estimatedDuration': interview_duration
        }
        
        logger.info(f"Mülakat paketi hazır: {package['packageId']}")
        
        return package
    
    def get_statistics(self) -> Dict:
        """API kullanım istatistiklerini döndür"""
        return {
            'total_generations': self.total_generations,
            'total_tokens_used': self.total_tokens_used,
            'average_tokens_per_generation': (
                self.total_tokens_used / self.total_generations 
                if self.total_generations > 0 else 0
            ),
            'model': self.model
        }
    
    async def test_connection(self) -> bool:
        """
        API bağlantısını test et.
        
        Basit bir tamamlama isteği göndererek API'nin çalıştığını doğrular.
        """
        try:
            logger.info("ChatGPT API bağlantısı test ediliyor...")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Say 'API connection successful' in exactly 3 words."}
                ],
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            
            if "successful" in result.lower():
                logger.info("✅ ChatGPT API bağlantısı başarılı!")
                return True
            else:
                logger.warning(f"Beklenmeyen yanıt: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ChatGPT API bağlantı hatası: {e}")
            return False


# Test için
if __name__ == "__main__":
    async def test():
        client = ChatGPTClient()
        
        # Bağlantı testi
        if await client.test_connection():
            print("✅ ChatGPT bağlantısı başarılı!")
            
            # Örnek iş ilanı üret
            print("\n📝 Örnek iş ilanı üretiliyor...")
            job_post = await client.generate_job_post(
                position_title="Full Stack Developer",
                seniority_level="senior",
                location="Istanbul, Turkey"
            )
            
            print(f"\nÜretilen İş İlanı:")
            print(f"  Pozisyon: {job_post['basicInfo']['title']}")
            print(f"  Şirket: {job_post['company']['name']}")
            print(f"  Lokasyon: {job_post['basicInfo']['location']}")
            print(f"  Deneyim: {job_post['requirements']['experience']['minimumYears']} yıl")
            
            # İstatistikler
            print(f"\nİstatistikler: {client.get_statistics()}")
        else:
            print("❌ ChatGPT bağlantısı başarısız!")
    
    asyncio.run(test())