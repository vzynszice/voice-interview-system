import asyncio
from typing import Union, AsyncGenerator
from pathlib import Path
from src.audio import AudioPlayer
import aiohttp
from loguru import logger
from elevenlabs import AsyncElevenLabs, VoiceSettings, Voice
import numpy as np
import soundfile as sf
from config import config

class ElevenLabsClient:
    def __init__(self):
        self.api_key = config.api.elevenlabs_api_key
        self.voice_id = config.api.elevenlabs_voice_id
        self.client = AsyncElevenLabs(api_key=self.api_key)
        self.voice_settings = VoiceSettings(
            stability=0.75,        
            similarity_boost=0.85, 
            style=0.5,            
            use_speaker_boost=True 
        )
        self.model_id = "eleven_multilingual_v2"  
        self.total_characters = 0
        self.total_requests = 0
        
        logger.info(f"ElevenLabs client has started. Voice ID: {self.voice_id}")
    
    async def text_to_speech(
        self,
        text: str,
        output_format = "wav",
        stream: bool = False
    ) -> Union[bytes, AsyncGenerator[bytes, None]]:
        try:
            logger.debug(f"Text is converting to speech. Length: {len(text)} characters")
            text = self._clean_text(text)
            if stream:
                audio_stream = await self.client.generate(
                    text=text,
                    voice=Voice(
                        voice_id=self.voice_id,
                        settings=self.voice_settings
                    ),
                    model=self.model_id,
                    stream=True,
                    output_format=output_format
                )
                
                return self._handle_stream(audio_stream)
            else:
                audio = await self.client.generate(
                    text=text,
                    voice=Voice(
                        voice_id=self.voice_id,
                        settings=self.voice_settings
                    ),
                    model=self.model_id,
                    stream=False,
                    output_format="pcm_16000"
                )
                if isinstance(audio, (bytes, bytearray)):
                    audio_bytes = audio
                else:                            
                    audio_bytes = b"".join([chunk async for chunk in audio])
                self.total_requests += 1
                self.total_characters += len(text)
                return audio_bytes
                
        except Exception as e:
            logger.error(f"ElevenLabs API error: {str(e)}")
            raise
    
    async def _handle_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in audio_stream:
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming hatası: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        import re
        text = re.sub(r'\s+', ' ', text)
        replacements = {
            '...': '.',     
            '!!': '!',      
            '??': '?',      
            '\n': ' ',      
            '\t': ' ',      
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.strip()
        max_length = 5000  
        if len(text) > max_length:
            logger.warning(f"Text is so large ({len(text)} characters), is decreasing...")
            text = text[:max_length-3] + "..."
        
        return text
    
    async def text_to_speech_file(
        self,
        text: str,
        output_path: Union[str, Path],
        output_format: str = "mp3_44100_128"
    ) -> Path:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_data = await self.text_to_speech(text, output_format, stream=False)
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Voice file is saved: {output_path}")
        return output_path
    
    async def convert_format(
        self,
        audio_data: bytes,
        from_format: str,
        to_format: str
    ) -> bytes:
        if from_format == to_format:
            return audio_data
        if from_format == "mp3" and to_format == "wav":
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tmp_mp3.write(audio_data)
                tmp_mp3.flush()
                return audio_data
        return audio_data
    
    async def get_voices(self) -> list:
        try:
            voices = await self.client.voices.get_all()
            
            voice_list = []
            for voice in voices.voices:
                voice_info = {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                    "labels": voice.labels,
                    "preview_url": voice.preview_url
                }
                voice_list.append(voice_info)
            logger.info(f"{len(voice_list)} voice file has found")
            return voice_list
            
        except Exception as e:
            logger.error(f"Voice list could not found: {e}")
            return []
    
    async def optimize_settings_for_interview(self):
        self.voice_settings = VoiceSettings(
            stability=0.85,        
            similarity_boost=0.75,  
            style=0.3,            
            use_speaker_boost=True
        )
        logger.info("Voice setting has optimized for the interview.")
    
    def calculate_cost(self, text: str) -> float:
        characters = len(text)
        cost_per_1000_chars = 0.22  
        estimated_cost = (characters / 1000) * cost_per_1000_chars
        return round(estimated_cost, 4)
    
    def get_statistics(self) -> dict:
        avg_chars = (
            self.total_characters / self.total_requests 
            if self.total_requests > 0 else 0
        )
        return {
            "total_requests": self.total_requests,
            "total_characters": self.total_characters,
            "average_characters_per_request": round(avg_chars, 1),
            "estimated_total_cost": f"${self.calculate_cost('') * self.total_requests:.2f}",
            "voice_id": self.voice_id,
            "model": self.model_id
        }
    
    async def test_connection(self) -> bool:
        try:
            logger.info("ElevenLabs API connection is checking...")
            test_text = "This is a test message."
            audio_data = await self.text_to_speech(
                text=test_text,
                output_format="mp3_44100_128",
                stream=False
            )
            if audio_data and len(audio_data) > 1000:  
                logger.info("✅ ElevenLabs API connection is successful!")
                return True
            else:
                logger.warning("Ses data is so small or None.")
                return False
                
        except Exception as e:
            logger.error(f"❌ ElevenLabs API connection error: {str(e)}")
            return False

if __name__ == "__main__":
    async def test():
        client = ElevenLabsClient()
        if await client.test_connection():
            print("✅ ElevenLabs connection is successful!")
            voices = await client.get_voices()
            if voices:
                print(f"\nUsable voices({len(voices)}):")
                for voice in voices[:3]:  
                    print(f"  - {voice['name']} ({voice['voice_id'][:8]}...)")
            
            sample_text = "This is an example interview question. Can you talk about your experience?"
            cost = client.calculate_cost(sample_text)
            print(f"\nExample text cost: ${cost:.4f}")
            print(f"\Statistics: {client.get_statistics()}")
        else:
            print("❌ ElevenLabs connection has failed!")
    asyncio.run(test())
