# 🌸 NIYRA — Personal AI Companion

> **A 100% local, privacy-first AI assistant inspired by J.A.R.V.I.S., built to think, remember, act, and grow with you.**

NIYRA is my vision for a modern personal AI companion that runs entirely on local hardware.  
It is designed to evolve from a simple voice assistant into a context-aware, action-oriented, always-available system that can understand me, remember me, help me manage my digital life, and keep everything private on my own PC.

The goal is not just to build a chatbot.  
The goal is to build a real assistant with memory, personality, automation, and awareness — something that feels alive, useful, and deeply personal.

---

## ✨ What NIYRA is meant to become

NIYRA is built around a simple idea:

- **Think locally**
- **Speak naturally**
- **Remember everything important**
- **Take actions on my behalf**
- **Stay private**
- **Grow in intelligence over time**

It is inspired by the J.A.R.V.I.S. concept, but adapted into something realistic, buildable, and useful in the real world.

---

## 🚀 Core Vision

NIYRA is designed to evolve into a companion that can:

- understand natural voice commands
- respond in a human-like way
- remember conversations and personal context
- control apps, files, and browser tasks
- communicate on my behalf
- detect my mood and respond accordingly
- run continuously in the background
- keep all data on-device

This project is my attempt to build a truly local AI assistant that feels personal, intelligent, and reliable.

---

## 🔒 Privacy-First Philosophy

NIYRA is built with one principle at the center:

**Everything stays on my machine.**

That means:

- no cloud dependency for core intelligence
- no external memory storage
- no unnecessary data leaks
- no third-party control over personal information
- no compromise on privacy

All voice processing, memory storage, reasoning, and automation are planned to remain local wherever possible.

---

## 🧠 What NIYRA will do

### Voice & Interaction
- Understand spoken language naturally
- Convert speech to text locally
- Speak back in a clear, human-like voice
- Handle context across multiple turns
- Support a conversational, companion-like tone

### Intelligence & Reasoning
- Solve tasks step by step
- Give opinions, suggestions, and pushback when needed
- Handle vague or incomplete commands
- Support both fast responses and deeper thinking

### Memory
- Remember conversations permanently
- Store personal history, goals, and preferences
- Recall technical notes, projects, and useful facts
- Keep memory persistent across restarts

### Information & Research
- Search the web when needed
- Read documents and extract information
- Scrape websites for data
- Monitor RSS/news sources locally

### PC Control
- Open apps and files
- Control mouse and keyboard
- Take screenshots
- Automate browser tasks
- Run scripts or commands when asked

### Communication
- Send emails
- Read and summarize inbox content
- Handle WhatsApp automation
- Manage reminders and calendar entries
- Generate responses while I am busy

### Awareness & Security
- Detect wake words locally
- Recognize my face before responding
- Detect emotion from voice and face
- Encrypt stored memory
- Run as a background assistant continuously

---

## 🛠️ Technology Stack

NIYRA follows a local-first stack built for privacy, performance, and modular growth.

| Layer | Technology | Purpose |
|------|------------|---------|
| **Brain (LLM)** | Ollama + LLaMA 3 | Deep local reasoning |
| **Fast Brain** | Phi-3 Mini via Ollama | Quick responses and lightweight reasoning |
| **Voice Input** | Whisper / faster-whisper | Local speech-to-text |
| **Voice Output** | pyttsx3 / Piper TTS | Offline text-to-speech |
| **Memory (Vector)** | ChromaDB | Semantic memory storage |
| **Memory (Facts)** | SQLite | Structured long-term facts |
| **Framework** | LangChain | Agent logic and tool routing |
| **API Layer** | FastAPI + WebSockets | Module communication and live interaction |
| **PC Control** | PyAutoGUI + subprocess | Mouse, keyboard, app control |
| **Browser Automation** | Playwright | Web navigation and browser actions |
| **Web Scraping** | BeautifulSoup | Extracting content from websites |
| **News Monitoring** | feedparser | RSS-based updates |
| **Email** | smtplib + imaplib | Local send/read email support |
| **WhatsApp** | whatsapp-web.js | Messaging automation |
| **Calendar** | ics library | Local reminders and scheduling |
| **Scheduling** | APScheduler | Background jobs and proactive tasks |
| **Wake Word** | Porcupine | Local wake word detection |
| **Vision** | OpenCV + DeepFace | Face and emotion analysis |
| **Face Recognition** | face_recognition | Identity-based access |
| **Emotion from Voice** | SpeechBrain | Mood detection from audio |
| **Security** | cryptography (Fernet) | Encrypted local storage |
| **Dashboard** | Streamlit | Visual memory/log viewer |

---

## 📅 Development Roadmap

NIYRA is planned in six phases, each building on the previous one.

### Phase 1 — Voice + Brain + Memory
**Goal:** make NIYRA hear, think, remember, and speak.

What I am building here:
- local speech input
- local speech output
- persistent memory
- local LLM reasoning
- the first working assistant loop

### Phase 2 — Personality + Speed
**Goal:** make NIYRA feel alive, fast, and personal.

What I am building here:
- custom personality
- streaming responses
- quick reply routing
- dual-brain structure for fast and deep thinking

### Phase 3 — PC Control + Actions
**Goal:** give NIYRA hands.

What I am building here:
- app launching
- file control
- screenshots
- browser automation
- web scraping
- tool-based action execution

### Phase 4 — Communication on My Behalf
**Goal:** make NIYRA act like a digital secretary.

What I am building here:
- email sending and reading
- inbox summarization
- WhatsApp automation
- calendar support
- reminders and automated replies

### Phase 5 — Awareness + Emotion + Security
**Goal:** make NIYRA more aware, more personal, and more secure.

What I am building here:
- wake-word activation
- voice emotion detection
- face recognition
- expression-based emotion reading
- local encryption for stored data

### Phase 6 — Always-On + Full Polish
**Goal:** make NIYRA a 24/7 assistant with a polished interface and background intelligence.

What I am building here:
- always-on background operation
- crash recovery
- auto-start on boot
- dashboard for memory and logs
- proactive suggestions and reminders

---

