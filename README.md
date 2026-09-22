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

**A local-first AI operating system — not a chatbot, a partner that thinks, remembers, and acts, while you keep the authority.**

![local-first](https://img.shields.io/badge/local--first-100%25-00ffc8?style=flat-square)
![privacy](https://img.shields.io/badge/privacy-structural-8a2be2?style=flat-square)
![orchestration](https://img.shields.io/badge/orchestration-Hermes%20Agent-3aa0ff?style=flat-square)
![status](https://img.shields.io/badge/status-work--in--progress-ff9f1c?style=flat-square)
![python](https://img.shields.io/badge/python-3.11%2B-yellow?style=flat-square)

</div>

---

```
$ niyra --boot

[BOOT] NIYRA v0.1 — initializing subsystems...

  [ OK ]  BRAIN ........... Gemma 4 E4B  (local, Ollama)
  [ OK ]  GUARD ........... risk classifier armed, fail-closed
  [ OK ]  VAULT ........... credential isolation active (handle-based)
  [ OK ]  MEMORY .......... ChromaDB + SQLite mounted
  [ OK ]  REGISTRY ........ tool registry online, write-locked to BRAIN
  [WARN]  SPEECH .......... implementation in progress — STT/TTS pending
  [ .. ]  OPERATOR ........ standing by, awaiting tool grants
  [ .. ]  RESEARCHER ...... cloud path scrubbed, standby

[READY] core loop online. awaiting input.
```

---

## `// 00 — THE ONE RULE`

> **The AI doesn't get unlimited control. I do.**

Every design decision in this system traces back to that line:

- ⚡ Sensitive actions never execute silently — explained, risk-scored, approved, *then* run.
- 🔐 Secrets never enter any agent's context — not even the one planning everything.
- 🧠 Locality ≠ trust. The Brain holds the *most* authority (it's the only agent that can create new tools/subagents) — which makes it the highest-value target for bad reasoning, not the safest place to skip the gate. It is checked exactly like everything else.

---

## `// 01 — ARCHITECTURE`

<div align="center">

```
          ┌─────────────────────────┐
          │         🧠 BRAIN         │
          │   plans · delegates     │
          │  sole tool-write power  │
          └────────────┬────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌───────────┐   ┌───────────┐   ┌───────────┐
  │ 🔎RESEARCH │   │ 💻OPERATOR │   │ 🎙️ SPEECH  │
  │  cloud-opt │   │ hands, no │   │  voice in │
  │  scrubbed  │   │ direct    │   │  & out    │
  │            │   │ writes    │   │           │
  └─────┬──────┘   └─────┬─────┘   └─────┬─────┘
        └────────────────┼───────────────┘
                          ▼
                  ┌──────────────┐
                  │  🛡️  GUARD    │
                  │ classify·gate │
                  │  ·log·audit   │
                  └───────┬──────┘
                          ▼
                  ┌──────────────┐
                  │  💾 MEMORY    │
                  │ local·shared │
                  │  ·compacted  │
                  └──────────────┘
```

</div>

| Module | Function | Notes |
|---|---|---|
| 🧠 **Brain** | Plans, decomposes, delegates | Only agent that may author new tools — and still runs through Guard |
| 🔎 **Researcher** | Web search, investigation | Cloud-optional; crosses out only through a scrub layer |
| 💻 **Operator** | PC, files, browser | Proposes changes — never applies directly |
| 🎙️ **Speech** | STT / TTS | **← currently being built** |
| 🛡️ **Guard** | Risk classification, approval gate, audit log | Fail-closed by default — unclassified = highest risk |
| 💾 **Memory** | Persistent context | Shared substrate every module reads/writes |

### Execution trace

```
INPUT   → me
PLAN    → Brain decomposes into tasks
DISPATCH→ Agents run (parallel where independent)
CHECK   → Guard classifies risk of each proposed action
GATE    → sensitive/reversible → approval requested → I decide
RUN     → approved action executes
LOG     → outcome + reasoning written to audit trail
STORE   → results committed to Memory
```

---

## `// 02 — SECURITY LAYER`

Structural guarantees, not prompt-level promises:

```
┌────────────────────────────────────────────────────────┐
│  MODEL REQUESTS →  "use_credential('email_password')"  │
│  MODEL RECEIVES ←  "<<CREDENTIAL:email_password>>"      │
│                     (opaque handle — never the secret)  │
│                                                          │
│  EXECUTOR RESOLVES HANDLE → real value, in-process only │
│  → used directly in the call → never re-enters context  │
└────────────────────────────────────────────────────────┘
```

- 🔒 **Blind credential injection** — a recognized pattern, not a custom invention; validated against prior art (PhantomKey, VaultKnox) rather than assumed novel.
- 🔑 **Write authority, hard-locked** — `propose_tool()` and its approval counterpart raise `PermissionError` for any caller other than Brain / Guard. Not a comment. A crash.
- 🌐 **Local/cloud boundary, enforced** — Brain, Guard, and Memory never cross out. Only Researcher (and Operator's coder-adjacent drafts) have a scrubbed, optional door.
- 📡 **Outbound scanning** — every agent response is checked for credential-shaped content *before* it can be spoken, displayed, or forwarded — not just tool call arguments.
- 🚫 **Fail-closed** — an unrecognized action defaults to the highest risk tier. Always.

---

## `// 03 — ORCHESTRATION: HERMES AGENT`

NIYRA runs on **Hermes Agent** rather than a hand-rolled runtime — an open-source, self-improving agent harness with native subagent orchestration, compacting memory, and a plugin/hook lifecycle. Guard hooks directly into that lifecycle:

| Hook | Guard's role there |
|---|---|
| `pre_tool_call` | Classify risk before anything dispatches |
| `pre_approval_request` | Inject reasoning + exact action into the approval prompt |
| `post_approval_response` | Log the decision — approved or denied |
| `post_tool_call` | Log the read-only path that skipped approval |
| `post_llm_call` | Scan outbound text for leaked credentials |

> Even Hermes's own team is candid that airtight secrets management is still an open problem on their end. Good reminder: "production-grade" ≠ "solved everywhere" — even for the infrastructure you build on top of.

---

## `// 04 — STACK`

<details>
<summary><strong>Expand full technology stack</strong></summary>

| Layer | Technology |
|---|---|
| Orchestration | Hermes Agent |
| Brain (local) | Gemma 4 E4B via Ollama |
| Fast path | Phi-3 Mini / Qwen2.5-3B |
| Coder (hybrid) | Qwen3-Coder-480B-A35B (cloud draft) + local apply |
| Researcher (cloud, optional) | NVIDIA API — Nemotron |
| Voice Input | Whisper / faster-whisper |
| Voice Output | Piper / Kokoro / pyttsx3 |
| Memory (vector) | ChromaDB |
| Memory (facts) | SQLite |
| Credential Vault | Fernet (cryptography) |
| PC Control | PyAutoGUI + subprocess |
| Browser Automation | Playwright (MCP) |
| Web Search | Tavily (MCP) |
| Scheduling | APScheduler |
| Wake Word | Porcupine |
| Vision *(later)* | OpenCV + DeepFace + face_recognition |
| Voice Emotion *(later)* | SpeechBrain |

</details>

**Hardware**

| Device | Role |
|---|---|
| PC — i5 11th Gen, 8GB RAM, RTX 2050 (4GB VRAM) | Brain, Guard, Operator, primary memory |
| Nothing Phone 3a Lite — Dimensity 7300 Pro, 8GB RAM | Offline fallback, wake-word/STT relay |

---

## `// 05 — STATUS`

```
BUILT & TESTED
  [██████████] risk classifier + credential vault + audit log
  [██████████] tool registry — write-locked, MCP vs custom
  [██████████] Hermes guardrail plugin — unit-tested vs mock context

IN PROGRESS
  [████░░░░░░] speech agent — STT/TTS integration        ← building now
  [██████░░░░] Hermes integration — against a live instance
  [███░░░░░░░] guardrail regression testing

QUEUED
  [░░░░░░░░░░] memory compaction
  [░░░░░░░░░░] tool registry expansion
  [░░░░░░░░░░] cross-device control (PC ↔ phone)
  [░░░░░░░░░░] Operator's browser/PC-control tool set
  [░░░░░░░░░░] Researcher's scrubbed cloud path
```

---

## `// 06 — ROADMAP`

| Phase | Goal |
|---|---|
| **1 — Voice + Brain + Memory** *(current)* | Local STT/TTS, persistent memory, the core Brain ↔ Guard ↔ Memory loop |
| **2 — Personality + Speed** | Custom tone, streaming responses, fast/deep dual routing |
| **3 — PC Control + Actions** | Operator gets real hands — always through the gate |
| **4 — Communication** | Email, calendar, messaging — approval-gated |
| **5 — Awareness + Emotion + Security** | Wake word, emotion detection, encrypted memory at rest |
| **6 — Always-On + Polish** | 24/7 operation, crash recovery, dashboard |

---

<div align="center">

**[github.com/maxtro64/Niyra-My_Cognitive_Partner](https://github.com/maxtro64/Niyra-My_Coginitive_Patner)**

*Still very much a work in progress — every claim above has a test or a cited source behind it, not just intention.*

`LOCAL` · `PRIVATE` · `CAPABLE` · `CONTROLLED`

</div>
