"""
Voice Agent Configuration
Optimized for RTX 2050 (4GB VRAM) + 8GB RAM
"""

import os
import sys

# Ensure UTF-8 stdout/stderr on Windows to handle emojis cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Limit OpenBLAS and OpenMP threads to prevent memory explosion on 8GB RAM systems
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ─── STT (Faster-Whisper) ────────────────────────────────────────────
STT_MODEL_SIZE = "base.en"      # Ultra-fast, accurate English model (~140MB)
STT_DEVICE = "cpu"              # CPU INT8 runs in <250ms, leaving 100% of 4GB VRAM for Ollama LLM
STT_COMPUTE_TYPE = "int8"       # 8-bit quantized for minimal CPU/RAM usage
STT_BEAM_SIZE = 1               # Greedy search for lowest latency
STT_LANGUAGE = "en"             # Lock to English for speed (skip language detection)

# ─── VAD (Silero) ────────────────────────────────────────────────────
VAD_THRESHOLD = 0.45            # Speech detection sensitivity (lower = more sensitive)
VAD_MIN_SPEECH_MS = 250         # Minimum speech duration to register (ms)
VAD_MIN_SILENCE_MS = 700        # Silence duration to consider end-of-speech (ms)
VAD_SPEECH_PAD_MS = 100         # Padding around speech segments (ms)

# ─── LLM (Ollama — Qwen2.5 0.5B / Phi3) ───────────────────────────────
LLM_MODEL = "qwen2.5:0.5b"      # Ultra-fast, conversational, fits seamlessly in low-RAM environments
LLM_BASE_URL = "http://localhost:11434"
LLM_SYSTEM_PROMPT = """You are Naiyra, a warm, witty, and helpful voice assistant. 
You respond in natural spoken English — short, conversational sentences.
Rules:
- Keep responses under 3 sentences unless asked for detail.
- Use contractions (I'm, you're, don't) for natural speech.
- Never use markdown, bullet points, or formatting — you are being spoken aloud.
- Show personality: be friendly, occasionally humorous, always helpful.
- If unsure, say so honestly rather than making things up."""

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 200            # Keep responses short for voice

# ─── TTS (Text-to-Speech) ─────────────────────────────────────────────
TTS_ENGINE = "edge"              # "edge" (0-RAM, ultra-natural neural voice) or "kokoro" (local 82M CPU)
TTS_VOICE = "en-US-AvaNeural"    # "en-US-AvaNeural" (edge) or "af_heart" (kokoro)
TTS_SPEED = 1.05                 # Slightly faster for natural feel
TTS_SAMPLE_RATE = 24000          # Native sample rate

# ─── Audio I/O ────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000       # Whisper expects 16kHz
AUDIO_CHANNELS = 1              # Mono
AUDIO_BLOCKSIZE = 512           # Audio chunk size for real-time processing
