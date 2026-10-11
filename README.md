<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│                                                           │
│    ███╗   ██╗██╗██╗   ██╗██████╗  █████╗                │
│    ████╗  ██║██║╚██╗ ██╔╝██╔══██╗██╔══██╗               │
│    ██╔██╗ ██║██║ ╚████╔╝ ██████╔╝███████║               │
│    ██║╚██╗██║██║  ╚██╔╝  ██╔══██╗██╔══██║               │
│    ██║ ╚████║██║   ██║   ██║  ██║██║  ██║               │
│    ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝               │
│                                                           │
│        M Y   C O G N I T I V E   P A R T N E R          │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**A local-first, multi-agent AI operating system that thinks, plans, speaks, and acts — while keeping you in total control.**

![local-first](https://img.shields.io/badge/local--first-100%25-00ffc8?style=flat-square)
![architecture](https://img.shields.io/badge/architecture-Multi--Agent%20Supervisor-3aa0ff?style=flat-square)
![interface](https://img.shields.io/badge/interface-Batcomputer%20Textual%20TUI-ffd700?style=flat-square)
![browser](https://img.shields.io/badge/browser-Playwright%20MCP-45ba4b?style=flat-square)
![privacy](https://img.shields.io/badge/privacy-structural-8a2be2?style=flat-square)
![hardware](https://img.shields.io/badge/hardware-RTX%202050%20%7C%208GB%20RAM-ff4757?style=flat-square)
![python](https://img.shields.io/badge/python-3.11%2B-yellow?style=flat-square)

</div>

---

```
$ python voice_agent/bruce.py --boot

[BOOT] NIYRA // WAYNE ENTERPRISES DEFENSE NETWORK — initializing subsystems...

  [ OK ]  SUPERVISOR ...... Multi-agent routing engine armed (dispatch map online)
  [ OK ]  ROUTER .......... LLMRouter & LayaGate low-latency classifier online
  [ OK ]  BATCOMPUTER TUI . Textual interactive terminal loaded (mouse & buttons active)
  [ OK ]  ACOUSTIC SENSOR . Faster-Whisper INT8 + Silero VAD v5 + Windows pycaw unmuted
  [ OK ]  VOICE SYNTH ..... Kokoro-82M neural voice / Edge-TTS AvaNeural streaming
  [ OK ]  BROWSER AGENT ... Playwright MCP connected (headed, visible window autofocus)
  [ OK ]  CODE SANDBOX .... CodeAct Docker isolation runtime standby
  [ OK ]  MESSAGE GATEWAY . Telegram, WhatsApp, Email adapters + APScheduler cron ready

[READY] All subsystems operational. Awaiting orders (speak or type).
```

---

## `// 00 — THE ONE RULE`

> **The AI doesn't get unlimited control. I do.**

Every design decision in Naiyra traces back to that principle:

- ⚡ **Sensitive actions never execute silently** — explained, risk-scored, approved, *then* run.
- 🔐 **Zero secret leakage** — credentials never enter any LLM's context window; handle-based resolution executes in-process only.
- 🛡️ **Locality ≠ blind trust** — the central supervisor holds planning authority, which makes it a high-value audit target. Every tool call and worker delegation is guarded and logged.
- 🚫 **Fail-closed default** — unclassified or ambiguous operations halt immediately for confirmation.

---

## `// 01 — MULTI-AGENT ARCHITECTURE`

Naiyra is structured as a **Supervisor-Worker multi-agent system**. The central Supervisor decomposes complex requests, evaluates confidence, and delegates subtasks to specialized worker agents via an explicit dispatch map.

<div align="center">

```
                      ┌──────────────────────────────────────────────┐
                      │             👤 USER INTERACTION              │
                      │  🎙️ Acoustic Sensor (Mic)  ·  ⌨️ Batcomputer TUI │
                      │       📱 Omnichannel Gateway (Telegram/Cron) │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │             🧠 SUPERVISOR AGENT              │
                      │    LLMRouter · Confidence Scoring · LayaGate │
                      └──────────────────────┬───────────────────────┘
                                             │
        ┌────────────────────┬───────────────┼───────────────┬────────────────────┐
        ▼                    ▼               ▼               ▼                    ▼
  ┌───────────┐        ┌───────────┐   ┌───────────┐   ┌───────────┐        ┌───────────┐
  │ 🌐 WEB     │        │ 💻 CODE   │   │ 🔎 SEARCH │   │ 📚 RAG    │        │ 🎙️ BRUCE   │
  │ Playwright│        │ Docker    │   │ Tavily    │   │ Local Docs│        │ Real-time │
  │ MCP Headed│        │ Sandbox   │   │ Web Intel │   │ & Memory  │        │ Conversat.│
  └─────┬─────┘        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘        └─────┬─────┘
        └────────────────────┴───────────────┼───────────────┴────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          🛡️ GUARD & SECURITY GATE             │
                      │  Blind Credential Vault · Outbound Redaction  │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │              💾 UNIFIED MEMORY               │
                      │   ChromaDB Vectors · SQLite Facts · File Log  │
                      └──────────────────────────────────────────────┘
```

</div>

### Specialized Worker Agents

| Agent | Subsystem | Capabilities & Mechanics |
|---|---|---|
| 🧠 **Supervisor** | `core/supervisor_agent.py` | Routes intent via `LLMRouter` with structured `RoutingDecision` outputs, self-contained subtask delegation, and fallback handling. |
| 🌐 **Web Agent** | `web_agent/web_agent.py` | Autonomous LangGraph browser agent using Playwright MCP (`@playwright/mcp`). Runs in a **visible headed browser** on screen, automatically restores and brings browser window to the foreground (`bring_browser_to_front`), extracts findings, verifies prices in INR, and protects against error streaks and loops. |
| 💻 **Code Agent** | `coding_agent/core.py` | Sandboxed CodeAct Python execution engine. Runs in an isolated Docker container (`--network none`, memory/CPU caps, read-only root) with fallback to `SubprocessSandbox`. Writes, tests, executes, and iterates on code until solved. |
| 📊 **Data Analyst** | `core/supervisor_agent.py` | Analyzes numerical and tabular datasets, runs computational pipelines, and extracts metrics inside the execution sandbox. |
| 🔎 **Web Search** | `core/tools.py` | Real-time live web intelligence powered by Tavily Search API with grounded factual synthesis. |
| 📚 **RAG Agent** | `core/supervisor_agent.py` | Semantic document retrieval across project documentation, architecture logs, and persistent memory files. |
| 🎙️ **Bruce (General)** | `voice_agent/bruce.py` | Tactical, sub-second conversational agent sporting a gritty Batman/Bruce Wayne persona. Streams responses sentence-by-sentence in real time. |

---

## `// 02 — BATCOMPUTER TUI (HERMES-STYLE INTERFACE)`

Naiyra features an interactive **Batcomputer Terminal User Interface (TUI)** built on [Textual](https://textual.textualize.io/) and [Rich](https://rich.readthedocs.io/), engineered after the Nous Research Hermes Agent CLI philosophy:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ═══ WAYNE ENTERPRISES DEFENSE NETWORK // BATCOMPUTER OS v2.4 (HERMES TUI) ════════════ │
│                                                                                        │
│        _==/       i   i       \==_          Active Toolsets                            │
│      /XX/         |\_/|         \XX\          Browser Tools: navigate, click, type...  │
│     /XXXX\         | |         /XXXX\         Intelligence:  tavily_web_search (live)  │
│    |XXXXXX\_      _--_       _/XXXXXX|                                                 │
│     XXXXXXXXxx\  /    \  /xxXXXXXXXX        Connected MCP Servers                      │
│      XXXXXXXXXX\/      \/XXXXXXXXXX           ● playwright (browser automation tools)  │
│           "~~~"          "~~~"                ● tavily     (real-time web search)      │
│                                                                                        │
│    Operative:   Bruce Wayne (The Dark Knight) Gotham Defense Protocols                 │
│    Cognitive:   Llama 3.2 (Ollama, local)       Acoustic Sensor: LISTENING             │
│    Voice Synth: Kokoro-82M (bane_voice.pt)      Voice abort:     ARMED                 │
│                                                                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ 🧪 Test Mic ] [ 🎙️ Unmute Mic ] [ ⚙️ Tools (25) ] [ ⚡ Subsystems ] [ ❓ Help ] [ 🛑 Abort ] │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ STATUS: LISTENING                                                                      │
│ 🦇 citizen@batcomputer > Click a button, enter query, or speak into microphone...       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key TUI Highlights

- 🖱️ **Full Mouse Support & Clickable Buttons**: Click buttons directly in your terminal to test microphone hardware, un-mute Windows audio, list categorized MCP tools, inspect subsystem health, or abort running operations.
- ⚡ **Dual-Mode Operation**: Talk naturally into the microphone or type directly into the tactical command prompt simultaneously.
- 📜 **Real-Time Typewriter Streaming**: Responses stream sentence-by-sentence into the central log as they are being thought, perfectly synchronized with TTS audio synthesis.
- 🛠️ **Subsystem Telemetry**: Hermes-style status reporting with `◆` sections and `✓ / ✗` diagnostic verification.
- 🔊 **Windows Audio Integration**: Built-in Windows Core Audio controller (`pycaw`) that detects muted microphones on startup and automatically unmutes and sets input levels to 100%.

---

## `// 03 — LOW-LATENCY VOICE & HARDWARE OPTIMIZATION`

Engineered to run **100% locally with $0 API costs** on modest consumer hardware:
- **Target Spec**: Intel Core i5 (11th Gen), 8 GB RAM, NVIDIA RTX 2050 Mobile (4 GB VRAM).
- **VRAM Preservation**: Faster-Whisper and Silero VAD are quantized and locked to CPU (INT8), leaving **100% of the 4 GB GPU VRAM dedicated to Ollama LLMs**.

```
┌──────────────────┐       ┌──────────────────┐       ┌────────────────────┐
│    Microphone    │ ───▶  │    Silero VAD    │ ───▶  │   Faster-Whisper   │
│  (sounddevice)   │       │   (PyTorch CPU)  │       │     (INT8 CPU)     │
└──────────────────┘       └──────────────────┘       └────────────────────┘
                                                                 │
                                                                 ▼
┌──────────────────┐       ┌──────────────────┐       ┌────────────────────┐
│  Speaker Output  │ ◀───  │ Kokoro/Edge-TTS  │ ◀───  │  Supervisor / LLM  │
│  (sounddevice)   │       │ (Streaming Synth)│       │ (Ollama Streaming) │
└──────────────────┘       └──────────────────┘       └────────────────────┘
```

- 🎙️ **Acoustic Sensor**: Non-blocking circular audio buffer (`voice_agent/audio_io.py`) with dynamic noise floor estimation and speech boundary tracking.
- 🗣️ **Dual-Engine Speech Synthesis**:
  - **Kokoro-82M**: Lightweight local neural TTS with custom voice cloning weights ([`assets/bane_voice.pt`](file:///c:/Personal/Python/Naiyraaa/assets/bane_voice.pt)).
  - **Microsoft Edge-TTS**: 0-RAM streaming fallback (`en-US-AvaNeural` / `en-US-ChristopherNeural`).
- 🛑 **Live Voice Barge-In**: Interrupt the agent mid-speech by saying *"stop"*, *"cancel"*, *"abort"*, or *"Bruce"* to instantly terminate playback and tool execution.
- ⚙️ **Thread Clamping**: Prevents Intel MKL allocator crashes and pagefile memory thrashing on Windows by capping OpenBLAS, OMP, and MKL threads to 1.

---

## `// 04 — OMNICHANNEL GATEWAY & PERSISTENT CRON`

Naiyra includes an **Omnichannel Message Gateway** (`message_gateway/`) that exposes the multi-agent supervisor to external communication platforms:

- 📱 **Platform Adapters**:
  - **Telegram Bot**: Full interactive chat interface with whitelist authorization (`telegram_adapter.py`).
  - **WhatsApp**: Webhook adapter ready for automated notifications (`whatsapp_adapter.py`).
  - **Email**: SMTP/IMAP bridge for processing incoming directives and dispatching status summaries (`email_adapter.py`).
- ⏰ **Universal Cron Scheduling**:
  - Built-in [APScheduler](https://apscheduler.readthedocs.io/) engine running persistent cron jobs saved to `cron_jobs.json`.
  - Schedule recurring prompts from any connected platform:
    ```bash
    # Schedule daily morning intelligence briefing at 8:00 AM
    /cron 0 8 * * * Generate my morning AI research brief and summarize top news

    # List all active recurring tasks
    /crons

    # Remove a scheduled task by ID
    /uncron a1b2c3d4
    ```

---

## `// 05 — TECHNOLOGY STACK`

<details>
<summary><strong>Expand full technology stack & dependencies</strong></summary>

| Subsystem | Technology | Role / Configuration | Hardware Footprint |
|---|---|---|---|
| **Supervisor & Routing** | `core/supervisor_agent.py` | LLMRouter, LayaGate (~30ms), LangChain | CPU / Minimal |
| **Browser Automation** | Playwright MCP (`@playwright/mcp`) | Headed Chromium/Edge, auto window foregrounding | ~150 MB RAM |
| **Code Execution** | Docker / Subprocess Sandbox | Sandboxed CodeAct environment (`--network none`) | Isolated container |
| **Terminal GUI (TUI)** | Textual + Rich | Clickable buttons, Hermes-style banner, streaming log | ~40 MB RAM |
| **Cognitive Brain (Local)** | Ollama (`llama3.2:3b` / `qwen2.5:0.5b`) | Streaming local generation with tool support | 2.2 GB VRAM / Fits 4GB |
| **Cognitive Brain (Cloud)**| NVIDIA NIM API (`Nemotron-3-Ultra`) | Heavy coding, research, and data analysis | $0 Cloud Tier |
| **Speech-to-Text (STT)** | Faster-Whisper (`base.en`) | CTranslate2 INT8 quantization, `beam_size=1` | CPU INT8 (~140 MB RAM) |
| **Voice Activity (VAD)** | Silero VAD v5 | PyTorch CPU, 512-chunk frames @ 16 kHz | CPU (~1ms per chunk) |
| **Voice Synthesis (TTS)** | Kokoro-82M / Edge-TTS | Bane custom embedding (`bane_voice.pt`) / AvaNeural | Local 82M CPU / 0-RAM |
| **Audio I/O & Windows** | `sounddevice` + `pycaw` | Async circular buffer, auto-unmute & gain adjustment | Zero-copy |
| **Web Intelligence** | Tavily Search API | Live internet search and grounded extraction | Remote API |
| **Gateway & Scheduler** | `python-telegram-bot` + APScheduler | Omnichannel routing & persistent cron jobs | Async background thread |

</details>

---

## `// 06 — PROJECT DIRECTORY STRUCTURE`

```
c:\Personal\Python\Naiyraaa\
│
├── core/                         # Core cognitive engine & multi-agent routing
│   ├── supervisor_agent.py       # Central supervisor coordinating worker dispatch map
│   ├── model_router.py           # Standalone LLM / Laya router with confidence scoring
│   ├── agent_config.py           # Model-to-agent mapping (Ollama vs NVIDIA NIM)
│   ├── laya_tools.py             # Single forward-pass decision gate & tool discovery
│   ├── tools.py                  # ToolManager, Playwright MCP, Tavily & window autofocus
│   └── config.py                 # Hardware-tuned config, thread limits & voice parameters
│
├── gui/                          # Batcomputer Terminal User Interface
│   └── main.py                   # Textual TUI with clickable buttons, hero banner & streaming
│
├── voice_agent/                  # Voice processing & interactive agents
│   ├── bruce.py                  # Tactical voice + TUI agent (Batman persona, barge-in)
│   ├── agent.py                  # LangGraph-based state machine (Plan-Execute-Verify)
│   ├── vad.py                    # Silero VAD v5 state machine with noise gating
│   ├── stt.py                    # Faster-Whisper INT8 CPU inference wrapper
│   ├── tts.py                    # Dual-engine TTS (Kokoro-82M + Edge-TTS)
│   ├── audio_io.py               # Non-blocking circular microphone buffer
│   └── test_agent.py             # Hardware-free & component diagnostic self-test suite
│
├── web_agent/                    # Autonomous browser navigation
│   ├── web_agent.py              # LangGraph browser agent with loop & error break guards
│   ├── common.py                 # Pydantic schemas (Finding, WebState, WebResult)
│   └── filter_response.py        # Token-saving observation compressor & summarizer
│
├── coding_agent/                 # Autonomous code generation & execution
│   ├── core.py                   # CodeAct loop (write, execute, observe, iterate)
│   ├── sandbox.py                # DockerSandbox (network-isolated) & SubprocessSandbox
│   └── workspace/                # Sandboxed persistent code workspace
│
├── message_gateway/              # Omnichannel platform integration
│   ├── core.py                   # Message bus & APScheduler persistent cron manager
│   ├── main.py                   # Gateway runner initializing Telegram, WhatsApp, Email
│   └── adapters/                 # Platform-specific adapters (telegram_adapter.py, etc.)
│
├── schemas/                      # Validated structured outputs
│   └── schema.py                 # Pydantic models (VoiceResponse, Plan, PlannedTask)
│
├── assets/                       # Custom neural voice embeddings
│   ├── bane_voice.pt             # PyTorch custom voice tensor
│   └── bane_voice.npy            # NumPy voice embedding
│
├── tests/                        # Integration and end-to-end test scenarios
│   └── test_morning_brief.py     # Complex multi-step morning intelligence briefing
│
├── .env.example                  # Environment configuration template
├── requirements.txt              # Unified Python dependency manifest
└── README.md                     # System documentation & technical reference
```

---

## `// 07 — QUICKSTART & USAGE`

### 0. Prerequisites & Environment Setup

Clone repository and create a Python 3.11+ virtual environment:

```powershell
# Clone the repository
git clone https://github.com/maxtro64/Niyra-My_Coginitive_Patner.git
cd Niyra-My_Coginitive_Patner

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure Playwright MCP browser packages are installed
npx -y @playwright/mcp --version
```

### 1. Configure Environment Secrets

Copy `.env.example` to `.env` and supply your API keys:

```powershell
cp .env.example .env
```

```env
# NVIDIA NIM Cloud API (for coding & deep reasoning)
NVIDIA_API_KEY="nvapi-..."

# Tavily API (for live web search)
TAVILY_API_KEY="tvly-..."

# Browser Automation (0 = visible headed window, 1 = headless)
PLAYWRIGHT_MCP_HEADLESS=0
HEADLESS=0
KEEP_BROWSER_OPEN=1

# (Optional) Messaging & Telegram Gateway
TELEGRAM_BOT_TOKEN="123456789:ABC..."
ALLOWED_USER_IDS="12345678"
```

### 2. Launching Modes

#### Option A: Launch Interactive Batcomputer TUI (Recommended)
Launches the full Textual terminal interface with mouse-clickable buttons, live telemetry, and direct supervisor execution:
```powershell
python gui/main.py
```

#### Option B: Launch Bruce Voice & Tactical Terminal Agent
Starts real-time continuous acoustic microphone listening + interactive command input with full Batman persona and voice barge-in:
```powershell
python voice_agent/bruce.py
```

#### Option C: Launch Omnichannel Gateway (Telegram / Cron)
Spawns background platform adapters and activates persistent cron scheduling:
```powershell
python message_gateway/main.py
```

#### Option D: Run Component Diagnostic Test Suite
Runs validation tests for logic, triggers, STT, VAD, LLM, TTS, and microphone hardware:
```powershell
python voice_agent/test_agent.py
```

---

## `// 08 — STATUS & MILESTONES`

```
BUILT & OPERATIONAL
  [██████████] Multi-Agent Supervisor with explicit dispatch map & LLMRouter
  [██████████] Playwright MCP headed browser agent with Windows autofocus
  [██████████] Sandboxed CodeAct Python coding engine (Docker + local fallback)
  [██████████] Batcomputer Textual TUI with clickable buttons & typewriter streaming
  [██████████] Low-latency voice pipeline (Faster-Whisper INT8 + Silero VAD + Kokoro TTS)
  [██████████] Windows Core Audio auto-unmute & volume calibration (pycaw)
  [██████████] Omnichannel message gateway & APScheduler persistent cron engine

IN PROGRESS
  [████████░░] Long-term memory vector compaction & graph memory integration
  [██████░░░░] Cross-device smartphone relay (wake-word STT bridge)

QUEUED
  [░░░░░░░░░░] Vision analysis module (OpenCV + screen perception)
  [░░░░░░░░░░] Voice emotion detection (SpeechBrain)
```

---

<div align="center">

**[github.com/maxtro64/Niyra-My_Cognitive_Partner](https://github.com/maxtro64/Niyra-My_Coginitive_Patner)**

`LOCAL` · `MULTI-AGENT` · `HEADED BROWSER` · `HERMES TUI` · `PRIVATE` · `CONTROLLED`

</div>
