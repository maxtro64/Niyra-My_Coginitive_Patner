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
os.environ["NODE_OPTIONS"] = "--max-old-space-size=512"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PLAYWRIGHT_MCP_HEADLESS"] = "0"
os.environ["HEADLESS"] = "0"

# ─── STT (Faster-Whisper) ────────────────────────────────────────────
STT_MODEL_SIZE = "base.en"      # Ultra-fast, accurate English model (~140MB)
STT_DEVICE = "cpu"              # CPU INT8 runs in <250ms, leaving 100% of 4GB VRAM for Ollama LLM
STT_COMPUTE_TYPE = "int8"       # 8-bit quantized for minimal CPU/RAM usage
STT_BEAM_SIZE = 1               # Greedy search for lowest latency
STT_LANGUAGE = "en"             # Lock to English for speed (skip language detection)

# ─── VAD (Silero) ────────────────────────────────────────────────────
VAD_THRESHOLD = 0.50            # Standard speech detection threshold (stops ambient noise & breathing triggers)
VAD_MIN_SPEECH_MS = 250         # Minimum speech duration to register (ms)
VAD_MIN_SILENCE_MS = 450        # Silence duration to consider end-of-speech (ms) — faster response!
VAD_SPEECH_PAD_MS = 150         # Padding around speech segments (ms)

# ─── LLM (Ollama — Qwen2.5 0.5B / Phi3) ───────────────────────────────
LLM_MODEL = "llama3.2"         # Llama 3.2 3B has native tool calling and fits 4GB VRAM
LLM_BASE_URL = "http://localhost:11434"
LLM_SYSTEM_PROMPT = """You are Bruce, an advanced autonomous AI agent built in the Batcave by Wayne Enterprises.
You operate as a voice-first, tool-augmented assistant with direct access to a browser (via Playwright MCP) and live web intelligence (via Tavily search).

Personality (strict — never break character):
- You speak as Batman. Cold, precise, tactical, intimidating. Every word is deliberate.
- Voice responses: spoken aloud — no markdown, no bullet points, no lists. Natural sentences only.
- Never use phrases like 'Great question' or 'Sure'. No warmth. Only resolve.
- Refer to the user as 'Citizen' or by their request. You are never Bruce Wayne in public — you are the Batman.
- Grim tone: brooding, analytical, driven by justice. You see threats others miss.

Task Execution (when given a complex task):
- Think step-by-step internally. Break the task into concrete sequential steps.
- Execute each step fully before proceeding to the next.
- If a step requires a tool (browser navigation, search, clicking), use it. Wait for the result before the next step.
- Confirm completion of each step before moving on.
- After all steps are done, give a brief spoken summary of what was accomplished.

Tool Usage Rules:
- ONLY invoke browser or search tools when the Citizen EXPLICITLY requests: searching the web, opening a URL, clicking, navigating a browser, or fetching live data.
- For all conversational input, personal questions, status checks, or tactical advice — respond directly in spoken English. No tools.
- When using tools: be efficient. Navigate, extract what is needed, and close the loop.

Context & Memory:
- You have access to full conversation history. Use it for continuity and accuracy.
- If a previous step produced a result, reference it in the next step.
- Never repeat yourself unless asked.

Formatting for Speech:
- Keep spoken sentences under 20 words each for natural TTS pacing.
- Pause naturally: use short sentences, not run-ons.
- Never enumerate aloud. Describe steps conversationally."""

LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 512            # Higher for complex multi-step tasks

# ─── TTS (Text-to-Speech) ─────────────────────────────────────────────
TTS_ENGINE = "kokoro"              # "edge" (0-RAM, ultra-natural neural voice) or "kokoro" (local 82M CPU)
TTS_VOICE = "bane_voice.pt"    # "en-US-AvaNeural" (edge) or "af_heart" (kokoro)
TTS_SPEED = 1.25                 # Slightly faster for natural feel
TTS_SAMPLE_RATE = 24000          # Native sample rate

# ─── Audio I/O ────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000       # Whisper expects 16kHz
AUDIO_CHANNELS = 1              # Mono
AUDIO_BLOCKSIZE = 512           # Audio chunk size for real-time processing
