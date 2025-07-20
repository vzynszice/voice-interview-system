# src/orchestrator/orchestrator.py (Tam, Eksiksiz ve Düzeltilmiş Hali)

import asyncio
import time
import subprocess
import json
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pydantic import BaseModel, Field

from config import config
from src.api_clients.groq_client import GroqClient
from src.api_clients.whisper_client import WhisperClient
from src.api_clients.elevenlabs_client import ElevenLabsClient
from src.api_clients.gct_client import GCTClient
from src.audio.audio_recorder import AudioRecorder
from src.audio.audio_player import AudioPlayer
from src.audio.screen_recorder import ScreenRecorder
from src.orchestrator.schema import JobInfo, CandidateInfo, QA


class InterviewOrchestrator:
    def __init__(self, job: JobInfo, candidate: CandidateInfo):
        self.job_info = job
        self.candidate_info = candidate
        self.config = config
        self.console = Console()
        self.phase: str = "warmup"
        self.transcript: List[QA] = []
        self.audio_segments = []
        self.start_time = datetime.now()
        self.output_filename_base = f"interview_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.question_count = 0
        self.interview_start_timestamp = None
        self.interview_end_timestamp = None
        self._setup_clients()

    def _setup_clients(self):
        """Tüm yardımcı API istemcilerini başlatır."""
        logger.info("Tüm istemciler başlatılıyor...")
        self.groq_client = GroqClient()
        self.whisper_client = WhisperClient()
        self.tts_client = ElevenLabsClient()
        self.gct_client = GCTClient()
        self.audio_recorder = AudioRecorder()
        self.audio_player = AudioPlayer()
        self.screen_recorder = ScreenRecorder(mic=False)

    def _create_question_prompt(self) -> List[dict]:
        """LLM'e soru üretmesi için daha basit bir yönerge hazırlar."""
        job_title = self.job_info.title
        skills = ', '.join(self.candidate_info.key_skills)
        previous_questions = [qa.text for qa in self.transcript if qa.role == "ai"]
        phase_prompts = {
            "warmup": "Ask a warm, friendly introductory question to make the candidate comfortable.",
            "technical": f"Ask a technical question about {job_title} skills, especially {skills}.",
            "behavioral": "Ask a behavioral question using the STAR method about past experiences.",
            "situational": f"Present a hypothetical scenario related to {job_title} work.",
            "closing": "Ask if they have any questions about the role or company."
        }
        base_prompt = phase_prompts.get(self.phase, "Ask a relevant interview question.")
        system_prompt = "You are an expert technical interviewer. Generate ONE clear, specific interview question. Return ONLY the question text, nothing else."
        user_prompt = f"""Context:
- Position: {job_title}
- Candidate skills: {skills}
- Interview phase: {self.phase}
- Question number: {self.question_count + 1}
Previous questions asked: {', '.join(previous_questions[-3:]) if previous_questions else 'None yet'}
Task: {base_prompt}
Important: Ask a DIFFERENT question than the ones already asked."""
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def _ask_ai_question(self) -> str:
        """LLM'den basit bir string soru al."""
        messages = self._create_question_prompt()
        try:
            response_text = await self.groq_client.generate_response(messages, temperature=0.8, max_tokens=150)
            question = response_text.strip().replace("\"", "") # Tırnak işaretlerini temizle
            if not question.endswith('?'):
                question += '?'
            logger.info(f"Üretilen soru: {question}")
            return question
        except Exception as e:
            logger.error(f"Soru üretme hatası: {e}")
            fallback_questions = {
                "warmup": ["Could you briefly introduce yourself...?"],
                "technical": ["Can you describe a challenging technical problem you've solved recently?"],
                "behavioral": ["Can you describe a time when you had to work under pressure?"],
                "situational": ["How would you approach debugging a production issue?"],
                "closing": ["Do you have any questions for me?"]
            }
            phase_questions = fallback_questions.get(self.phase, fallback_questions["technical"])
            import random
            return random.choice(phase_questions)

    async def _speak(self, tr_text: str):
        """Metni seslendir ve timestamp'le kaydet"""
        try:
            audio_data = await self.tts_client.text_to_speech(tr_text, output_format="pcm_16000")
            if audio_data:
                timestamp = time.time() - self.interview_start_timestamp
                play_start = time.time()
                await self.audio_player.play(audio_data, sample_rate=16000)
                play_duration = time.time() - play_start
                self.audio_segments.append({'data': audio_data, 'timestamp': timestamp, 'duration': play_duration, 'type': 'ai_speech'})
        except Exception as e:
            logger.error(f"Seslendirme hatası: {str(e)}")

    async def _listen(self) -> str:
        """Kullanıcıyı dinle ve timestamp'le kaydet"""
        try:
            record_start = time.time()
            timestamp = record_start - self.interview_start_timestamp
            wav_data = await self.audio_recorder.start_recording()
            if not wav_data or len(wav_data) < 2000:
                return ""
            record_duration = time.time() - record_start
            self.audio_segments.append({'data': wav_data, 'timestamp': timestamp, 'duration': record_duration, 'type': 'human_speech'})
            return await self.whisper_client.transcribe_audio(wav_data)
        except Exception as e:
            logger.error(f"Dinleme hatası: {str(e)}")
            return "[HATA: Yanıt işlenemedi]"

    async def _translate(self, text: str, src: str, tgt: str) -> str:
        if not text or not text.strip():
            return ""
        try:
            return await self.gct_client.translate(text, source_lang=src, target_lang=tgt)
        except Exception as e:
            logger.error(f"Çeviri hatası: {e}")
            return text

    def _create_synchronized_audio(self) -> Optional[Path]:
        """Tüm ses segmentlerini timestamp'lere göre birleştir"""
        if not self.audio_segments:
            return None
        try:
            total_duration = max(seg['timestamp'] + seg['duration'] for seg in self.audio_segments)
            sample_rate = 16000
            num_samples = int(total_duration * sample_rate)
            audio_buffer = np.zeros(num_samples, dtype=np.int16)
            for segment in self.audio_segments:
                start_sample = int(segment['timestamp'] * sample_rate)
                audio_data = segment['data'][44:] if segment['type'] == 'human_speech' else segment['data']
                segment_array = np.frombuffer(audio_data, dtype=np.int16)
                end_sample = start_sample + len(segment_array)
                if end_sample <= len(audio_buffer):
                    audio_buffer[start_sample:end_sample] += segment_array
                else:
                    valid_part = segment_array[:len(audio_buffer) - start_sample]
                    audio_buffer[start_sample:] += valid_part
            audio_path = self.config.recording.recording_output_dir / f"{self.output_filename_base}_audio.wav"
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(audio_buffer.tobytes())
            return audio_path
        except Exception as e:
            logger.error(f"Ses birleştirme hatası: {e}")
            return None

    async def _mux_video_and_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Video ve sesi FFmpeg ile birleştirir."""
        logger.info(f"Video ve ses birleştiriliyor -> {output_path}")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(output_path)]
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f"FFmpeg hatası: {stderr.decode(errors='ignore')}")
            else:
                logger.info(f"✅ Video başarıyla oluşturuldu: {output_path}")
        except Exception as e:
            logger.error(f"Muxing işlemi başarısız: {e}")

    async def run(self):
        """Ana mülakat döngüsü."""
        temp_video_path = self.config.recording.recording_output_dir / f"temp_{self.output_filename_base}.mp4"
        self.interview_start_timestamp = time.time()
        max_duration = self.config.app.max_interview_duration * 60
        rec_task = asyncio.create_task(self.screen_recorder.record(output_path=temp_video_path, max_duration=max_duration + 10))
        logger.info(f"📹 Video kaydı başlatıldı -> {temp_video_path}")

        try:
            for ph in self.config.interview.interview_phases:
                self.phase = ph
                num_questions = self.config.interview.questions_per_phase.get(ph, 1)
                self.console.print(Panel(f"[bold cyan]Faz: {ph.capitalize()} ({num_questions} Soru)[/bold cyan]"))
                for i in range(num_questions):
                    self.question_count += 1
                    self.console.print(f"\n[yellow]Soru {i+1}/{num_questions} hazırlanıyor...[/yellow]")
                    en_question = await self._ask_ai_question()
                    tr_question = await self._translate(en_question, "en", "tr")
                    self.console.print(Text(f"🤖 AI: {tr_question}", style="bright_blue"))
                    await self._speak(tr_question)
                    self.console.print(Text("️💬 Lütfen yanıtlayın...", style="dim"))
                    tr_answer = await self._listen()
                    self.console.print(Text(f"👤 Siz: {tr_answer or '[Sessizlik]'}", style="bright_green"))
                    en_answer = await self._translate(tr_answer, "tr", "en")
                    ts = time.time()
                    self.transcript.append(QA(role="ai", text=en_question, ts=ts))
                    self.transcript.append(QA(role="human", text=en_answer, ts=ts))
                    logger.info(f"✅ Q&A {i+1} tamamlandı. Faz: {self.phase}")
            self.console.print(Panel("[bold green]Mülakat tamamlandı. Son işlemler yapılıyor...[/bold green]"))
        finally:
            self.interview_end_timestamp = time.time()
            interview_duration = self.interview_end_timestamp - self.interview_start_timestamp
            
            logger.info("Video kaydı sonlandırılıyor...")
            await asyncio.sleep(2)
            self.screen_recorder.stop()
            try:
                await asyncio.wait_for(rec_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Video kaydı zaman aşımına uğradı.")
            
            audio_path = self._create_synchronized_audio()
            
            if temp_video_path.exists() and audio_path and audio_path.exists():
                final_video_path = self.config.recording.recording_output_dir / f"{self.output_filename_base}.mp4"
                await self._mux_video_and_audio(temp_video_path, audio_path, final_video_path)
                
                # Dosya boyutlarını logla ve TEMİZLİK BURADA YAPILIYOR
                video_size = temp_video_path.stat().st_size / 1024 / 1024
                audio_size = audio_path.stat().st_size / 1024 / 1024
                logger.info(f"📊 Geçici dosya boyutları - Video: {video_size:.1f}MB, Ses: {audio_size:.1f}MB")
                temp_video_path.unlink()
                audio_path.unlink()
                
            elif temp_video_path.exists():
                final_video_path = self.config.recording.recording_output_dir / f"{self.output_filename_base}_silent.mp4"
                temp_video_path.rename(final_video_path)
            
            transcript_output_path = self.config.app.transcript_dir / f"{self.output_filename_base}.jsonl"
            with transcript_output_path.open("w", encoding="utf-8") as f:
                metadata = {"interview_duration": interview_duration}
                f.write(json.dumps(metadata) + "\n")
                for qa in self.transcript: f.write(qa.model_dump_json() + "\n")
            logger.info(f"📄 Transkript kaydedildi -> {transcript_output_path}")