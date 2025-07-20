# Voice Interview System

Multilingual native voice interviews: Microphone → STT → (Translation) → LLM → TTS narration.

## New Features
- **Local LLM:** Ollama (Gemma3:1b)
- **STT:** Whisper / turbo (local)
- **TTS:** pyttsx3 (offline)
- **Translation:** Argos Translate (offline)
- **Orchestrator:** Modular state + error/fallback layer
- **VAD:** webrtcvad (configurable)

## Before
- **Local LLM:** Groq / ChatGPT (single remote)
- **STT:** Whisper API (remote)
- **TTS:** ElevenLabs (online)
- **Translation:** Google Cloud Translation (GCT)
- **Orchestrator:** Procedural (no central state/fallback)
- **VAD:** webrtcvad (hard‑coded)


## Quick Start
```bash
git clone https://github.com/vzynszice/voice-interview-system.git
cd voice-interview-system
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
python main.py


## .env
OPENAI_API_KEY=
GROQ_API_KEY=
ELEVEN_API_KEY=
GCT_API_KEY=
WHISPER_MODEL=base
OLLAMA_MODEL=gemma3:1b
PRIMARY_LANGUAGE=tr
TARGET_LANGUAGE=en
