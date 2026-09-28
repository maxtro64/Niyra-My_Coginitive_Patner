"""
Naiyra Voice Agent — LangGraph Execution Graph
VAD → STT → [Plan → Execute → Verify → Retry] → TTS

Graph nodes:
  listen       : Wait for speech from VAD/mic
  transcribe   : Whisper STT → user text
  plan         : LLM decomposes task into steps (or marks as direct)
  execute_step : LLM executes current step with browser tools
  verify_step  : LLM + browser_snapshot verify step completion
  retry_step   : Increment retry counter, modify prompt
  speak        : Stream sentences → TTS queue
  done         : Update history, reset state

Edge conditions:
  after verify: SUCCESS → next step (or done) | FAILURE + retries left → retry_step | exhausted → next step (mark failed)
  after plan:   steps=[] → execute_step directly | steps → execute_step[0]
  task fail threshold → re-plan (task retry)
"""

import threading
import time
import queue
import numpy as np
import sounddevice as sd
from typing import TypedDict, Annotated, List, Optional, Any
import operator

from langgraph.graph import StateGraph, END, START

from config import AUDIO_SAMPLE_RATE, AUDIO_BLOCKSIZE, AUDIO_CHANNELS
from vad import VoiceActivityDetector
from stt import SpeechToText
from llm import LanguageModel
from tts import TextToSpeech
from tools import SyncPlaywrightMcp

# ─── ANSI Colors ─────────────────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init()
    C_USER   = Fore.CYAN
    C_AGENT  = Fore.YELLOW
    C_SYS    = Fore.WHITE + Style.DIM
    C_TOOL   = Fore.GREEN
    C_ERR    = Fore.RED
    C_STEP   = Fore.MAGENTA
    C_OK     = Fore.GREEN + Style.BRIGHT
    C_FAIL   = Fore.RED + Style.BRIGHT
    C_RESET  = Style.RESET_ALL
    C_BOLD   = Style.BRIGHT
except ImportError:
    C_USER = C_AGENT = C_SYS = C_TOOL = C_ERR = C_STEP = C_OK = C_FAIL = C_RESET = C_BOLD = ""


# ─── Graph State ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Input
    user_text: str                      # The current user utterance

    # Planning
    steps: List[str]                    # Decomposed step list (empty = direct reply)
    step_index: int                     # Which step we're on
    step_retries: int                   # Retry count for current step
    task_retries: int                   # Full task restart count
    failed_steps: int                   # Steps that exhausted retries

    # Execution
    step_response: str                  # LLM spoken output for current step
    step_succeeded: bool                # Did verifier confirm success?
    sentences: Annotated[List[str], operator.add]  # Accumulates spoken sentences

    # Context
    shared_messages: List[dict]         # Persistent LLM message thread (browser session)
    interrupted: bool                   # Abort flag

    # Output
    full_response: str                  # Complete response for history


# ─── Constants ────────────────────────────────────────────────────────────────
MAX_STEP_RETRIES = 3
MAX_TASK_RETRIES = 2


# ─── Node Implementations ────────────────────────────────────────────────────

def node_plan(state: AgentState, llm: LanguageModel, mcp: Any) -> dict:
    """
    Plan node: decompose user_text into atomic steps.
    If it's a direct query, steps = [].
    """
    user_text = state["user_text"]

    if llm._is_complex_task(user_text):
        steps = llm._plan_task(user_text) or []
        if steps == ["respond directly"] or not steps:
            steps = []
        else:
            print(f"\n{C_STEP}  ◈ MISSION PLAN — {len(steps)} steps{C_RESET}")
            for i, s in enumerate(steps, 1):
                print(f"{C_STEP}    [{i}] {s}{C_RESET}")
    else:
        steps = []

    # Build the shared message thread (persists browser session across steps)
    from config import LLM_SYSTEM_PROMPT
    shared_messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    shared_messages.extend(llm.history[-llm._max_history:])

    if steps:
        plan_ctx = (
            f"The user has requested: {user_text}\n\n"
            f"Mission plan ({len(steps)} steps — execute in order):\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
            + "\n\nIMPORTANT: Browser state is preserved between steps."
        )
        shared_messages.append({"role": "system", "content": plan_ctx})

    return {
        "steps": steps,
        "step_index": 0,
        "step_retries": 0,
        "task_retries": state.get("task_retries", 0),
        "failed_steps": 0,
        "sentences": [],
        "shared_messages": shared_messages,
        "full_response": "",
        "step_response": "",
        "step_succeeded": False,
    }


def node_execute_step(state: AgentState, llm: LanguageModel, mcp: Any,
                      tts: TextToSpeech, interrupt_check) -> dict:
    """
    Execute node: run the current step (or full direct reply if no plan).
    Streams sentences to TTS in real-time.
    Uses the SHARED message thread so browser session context is preserved.
    """
    steps      = state["steps"]
    idx        = state["step_index"]
    retries    = state["step_retries"]
    shared_msgs = state["shared_messages"]
    user_text  = state["user_text"]

    enable_tools = mcp is not None and (
        steps or llm._is_complex_task(user_text)
    )
    llm_tools = mcp.convert_tools_for_llm() if enable_tools else None

    if not steps:
        # Direct reply — no plan
        step_label = "direct"
        prompt = user_text
        print(f"\n{C_AGENT}  ◈ RESPONDING DIRECTLY...{C_RESET}")
    else:
        step = steps[idx]
        step_label = f"step {idx+1}/{len(steps)}"
        if retries == 0:
            prompt = (
                f"Execute step {idx + 1} of {len(steps)}: {step}\n"
                f"Use browser tools if needed. Speak your result in 1-2 short sentences. "
                f"Browser session is persistent from previous steps."
            )
            print(f"\n{C_STEP}  ◈ EXECUTING [{idx+1}] {step}{C_RESET}")
        else:
            prompt = (
                f"RETRY {retries}/{MAX_STEP_RETRIES} — step {idx+1} was not verified.\n"
                f"Goal: {step}\n"
                f"Try differently: use browser_snapshot first, then more specific selectors. "
                f"Speak result in 1-2 sentences."
            )
            print(f"\n{C_STEP}  ↻ RETRY [{idx+1}] attempt {retries}/{MAX_STEP_RETRIES}{C_RESET}")

    # Stream from LLM using the shared message thread
    sentences_this_step: List[str] = []
    full_text = ""

    def _on_tool_call(name, args):
        print(f"{C_TOOL}    ▸ TOOL {name}{C_RESET}")

    def _on_tool_result(name, length):
        print(f"{C_TOOL}    ◂ {name} ({length} bytes){C_RESET}")

    print(f"\n{C_AGENT}  Bruce: {C_RESET}", end="", flush=True)

    for sentence in llm._execute_step_stream(
        shared_msgs, prompt, llm_tools,
        interrupt_check, _on_tool_call, _on_tool_result, mcp
    ):
        if interrupt_check():
            break
        if not sentence.strip():
            continue
        sentences_this_step.append(sentence)
        full_text += sentence + " "
        print(f"{sentence} ", end="", flush=True)
        tts.speak(sentence)

    print()  # newline after response

    return {
        "step_response": full_text.strip(),
        "sentences": sentences_this_step,
        "shared_messages": shared_msgs,   # pass back the mutated thread
        "full_response": state.get("full_response", "") + full_text,
    }


def node_verify_step(state: AgentState, llm: LanguageModel, mcp: Any) -> dict:
    """
    Verify node: check whether the current step succeeded.
    Uses browser_snapshot + spoken output for judgement.
    """
    steps    = state["steps"]
    idx      = state["step_index"]
    spoken   = state["step_response"]

    if not steps:
        # Direct response — always passes
        return {"step_succeeded": True}

    step = steps[idx]
    succeeded = llm._verify_step(step, spoken, mcp)

    if succeeded:
        print(f"{C_OK}  ✓ VERIFIED [{idx+1}]{C_RESET}")
    else:
        print(f"{C_FAIL}  ✗ VERIFY FAILED [{idx+1}]{C_RESET}")

    return {"step_succeeded": succeeded}


def node_advance_step(state: AgentState) -> dict:
    """Move to the next step — reset all per-step state."""
    new_idx = state["step_index"] + 1
    steps   = state["steps"]
    if new_idx < len(steps):
        print(f"{C_OK}  → NEXT STEP [{new_idx+1}/{len(steps)}]: {steps[new_idx]}{C_RESET}")
    return {
        "step_index":    new_idx,
        "step_retries":  0,
        "step_succeeded": False,   # ← must reset: stale True would skip verify next round
        "step_response": "",
    }


def node_retry_step(state: AgentState) -> dict:
    """Increment retry counter — clear stale success flag before re-executing."""
    return {
        "step_retries":   state["step_retries"] + 1,
        "step_succeeded": False,   # ← guard: stale True must not skip the retry verify
        "step_response":  "",
    }


def node_mark_step_failed(state: AgentState) -> dict:
    """Step exhausted retries — mark failed, move on."""
    idx = state["step_index"]
    print(f"{C_FAIL}  ✗ STEP [{idx+1}] EXHAUSTED — moving on{C_RESET}")
    return {
        "failed_steps": state["failed_steps"] + 1,
        "step_index": state["step_index"] + 1,
        "step_retries": 0,
    }


def node_replan(state: AgentState, llm: LanguageModel) -> dict:
    """
    Task retry: re-plan with failure context, restart step index.
    """
    attempt = state["task_retries"] + 1
    print(f"\n{C_ERR}  ↻ MISSION RESTART — re-planning (attempt {attempt}/{MAX_TASK_RETRIES}){C_RESET}")

    new_steps = llm._plan_task(
        f"{state['user_text']}\n\nPrevious attempt failed on multiple steps. "
        f"Re-plan with more specific, atomic steps using exact URLs and element descriptions."
    ) or state["steps"]

    print(f"{C_STEP}  ◈ NEW PLAN — {len(new_steps)} steps{C_RESET}")
    for i, s in enumerate(new_steps, 1):
        print(f"{C_STEP}    [{i}] {s}{C_RESET}")

    # Rebuild shared messages fresh (new browser context attempt)
    from config import LLM_SYSTEM_PROMPT
    shared_messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    shared_messages.extend(llm.history[-llm._max_history:])
    plan_ctx = (
        f"The user has requested: {state['user_text']}\n\n"
        f"Revised mission plan ({len(new_steps)} steps):\n"
        + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(new_steps))
    )
    shared_messages.append({"role": "system", "content": plan_ctx})

    return {
        "steps": new_steps,
        "step_index": 0,
        "step_retries": 0,
        "failed_steps": 0,
        "task_retries": attempt,
        "shared_messages": shared_messages,
    }


def node_finalize(state: AgentState, llm: LanguageModel) -> dict:
    """Update conversation history, return final state."""
    llm.history.append({"role": "user",      "content": state["user_text"]})
    llm.history.append({"role": "assistant", "content": state.get("full_response", "").strip()})
    if len(llm.history) > llm._max_history * 2:
        llm.history = llm.history[-llm._max_history * 2:]

    steps_done = len(state.get("steps", []))
    failed     = state.get("failed_steps", 0)
    if steps_done:
        print(f"\n{C_SYS}  ◈ Mission complete: {steps_done - failed}/{steps_done} steps verified.{C_RESET}")

    return {}


# ─── Edge Conditions ─────────────────────────────────────────────────────────

def edge_after_verify(state: AgentState) -> str:
    """Route after verification — SUCCESS goes straight to next step (no intermediate node)."""
    steps   = state["steps"]
    idx     = state["step_index"]
    retries = state["step_retries"]

    if not steps:
        # Direct reply — done
        return "finalize"

    if state["step_succeeded"]:
        # Step passed — next step or done (advance happens in node_advance_step)
        if idx + 1 >= len(steps):
            return "finalize"
        return "advance"
    else:
        # Step failed
        if retries < MAX_STEP_RETRIES:
            return "retry"
        return "mark_failed"


def edge_after_advance(state: AgentState) -> str:
    """After incrementing step_index, go execute or finalize."""
    if state["step_index"] >= len(state["steps"]):
        return "finalize"
    return "execute"


def edge_after_mark_failed(state: AgentState) -> str:
    """After marking a step failed, decide: continue or re-plan task."""
    failed = state["failed_steps"]
    total  = len(state["steps"])
    task_retries = state["task_retries"]

    # If more than half failed and we still have task retries left → re-plan
    if failed >= max(1, total // 2) and task_retries < MAX_TASK_RETRIES:
        return "replan"

    # Otherwise continue to next step or finalize
    if state["step_index"] >= total:
        return "finalize"
    return "execute"


def edge_after_replan(state: AgentState) -> str:
    """After re-planning, always go back to execute."""
    return "execute"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_agent_graph(llm: LanguageModel, mcp: Any, tts: TextToSpeech,
                      interrupt_check) -> Any:
    """
    Build and compile the LangGraph execution graph.

    Flow:
      plan → execute_step → verify_step → [advance | retry | mark_failed | replan] → finalize

    Edges:
      verify  → advance       : step succeeded, more steps remain
      verify  → finalize      : step succeeded or direct reply, no more steps
      verify  → retry         : step failed, retries remain
      verify  → mark_failed   : step failed, retries exhausted
      advance → execute       : more steps
      advance → finalize      : all steps done
      mark_failed → execute   : more steps, failures below threshold
      mark_failed → replan    : too many failures, task retries available
      mark_failed → finalize  : too many failures, no task retries left
      replan  → execute       : restart with new plan
    """
    g = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    g.add_node("plan",        lambda s: node_plan(s, llm, mcp))
    g.add_node("execute",     lambda s: node_execute_step(s, llm, mcp, tts, interrupt_check))
    g.add_node("verify",      lambda s: node_verify_step(s, llm, mcp))
    g.add_node("advance",     lambda s: node_advance_step(s))
    g.add_node("retry",       lambda s: node_retry_step(s))
    g.add_node("mark_failed", lambda s: node_mark_step_failed(s))
    g.add_node("replan",      lambda s: node_replan(s, llm))
    g.add_node("finalize",    lambda s: node_finalize(s, llm))

    # ── Entry ───────────────────────────────────────────────────────────────
    g.set_entry_point("plan")

    # ── Edges ───────────────────────────────────────────────────────────────
    g.add_edge("plan",    "execute")
    g.add_edge("execute", "verify")

    g.add_conditional_edges("verify", edge_after_verify, {
        "advance":  "advance",
        "retry":    "retry",
        "mark_failed": "mark_failed",
        "finalize": "finalize",
    })

    g.add_conditional_edges("advance", edge_after_advance, {
        "execute":  "execute",
        "finalize": "finalize",
    })

    # retry loops back to execute (with incremented step_retries)
    g.add_edge("retry", "execute")

    g.add_conditional_edges("mark_failed", edge_after_mark_failed, {
        "execute":  "execute",
        "replan":   "replan",
        "finalize": "finalize",
    })

    g.add_conditional_edges("replan", edge_after_replan, {
        "execute": "execute",
    })

    g.add_edge("finalize", END)

    return g.compile()


# ─── Voice Agent ──────────────────────────────────────────────────────────────

class NaiyraAgent:
    """
    Real-time voice agent with LangGraph execution graph.
    Microphone → Silero VAD → Faster-Whisper STT → LangGraph → TTS
    """

    def __init__(self):
        print(f"\n{C_BOLD}{'='*60}")
        print(f"  NAIYRA — LangGraph Voice Agent Initializing...")
        print(f"{'='*60}{C_RESET}\n")

        self.stt = SpeechToText()
        self.vad = VoiceActivityDetector()
        self.llm = LanguageModel()
        self.tts = TextToSpeech()

        print(f"  {C_SYS}Connecting to MCP / Browser...{C_RESET}")
        self.mcp = SyncPlaywrightMcp()
        try:
            self.mcp.connect()
            tools = self.mcp.get_tools()
            print(f"  {C_SYS}MCP online — {len(tools)} tools loaded.{C_RESET}")
        except Exception as e:
            print(f"  {C_ERR}MCP unavailable: {e}{C_RESET}")
            self.mcp = None

        # State
        self._is_running    = False
        self._is_processing = False
        self._interrupted   = False
        self._audio_stream  = None
        self._speech_chunks: List[np.ndarray] = []

        # TTS queue (speak sentences as they arrive)
        self._tts_queue: queue.Queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # Build LangGraph
        self._graph = build_agent_graph(
            self.llm, self.mcp, self.tts,
            interrupt_check=lambda: self._interrupted
        )

        print(f"\n{C_BOLD}{'='*60}")
        print(f"  ✓ All systems online. Speak or press Ctrl+C to quit.")
        print(f"{'='*60}{C_RESET}\n")

    # ─── TTS worker ───────────────────────────────────────────────────────────

    def _tts_worker(self):
        while True:
            try:
                item = self._tts_queue.get()
                if item is None:
                    break
                if not self._interrupted and item.strip():
                    self.tts.speak(item)
                self._tts_queue.task_done()
            except Exception:
                pass

    # ─── Audio pipeline ───────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy().astype(np.float32)
        chunk_size = 512
        for i in range(0, len(chunk), chunk_size):
            sub = chunk[i:i + chunk_size]
            if len(sub) < chunk_size:
                sub = np.pad(sub, (0, chunk_size - len(sub)))
            result = self.vad.process_chunk(sub)

            if result["event"] == "speech_start":
                self._speech_chunks = []

            if self.vad.is_speaking or result["event"] == "speech_end":
                self._speech_chunks.append(sub)

            if result["event"] == "speech_end" and not self._is_processing:
                speech = np.concatenate(self._speech_chunks)
                self._speech_chunks = []
                self.vad.reset()
                threading.Thread(
                    target=self._handle_speech, args=(speech,), daemon=True
                ).start()

    def _handle_speech(self, audio: np.ndarray):
        user_text = self.stt.transcribe(audio)
        if not user_text or len(user_text.strip()) < 2:
            return

        lower = user_text.lower()
        if self._is_processing:
            if any(w in lower for w in ["stop", "cancel", "abort"]):
                self._interrupted = True
                self.tts.stop()
            return

        print(f"\n  {C_BOLD}{C_USER}You:{C_RESET} {user_text}")
        self._run_graph(user_text)

    # ─── LangGraph invocation ─────────────────────────────────────────────────

    def _run_graph(self, user_text: str):
        """Invoke the LangGraph pipeline for a user utterance."""
        self._is_processing = True
        self._interrupted   = False

        initial_state: AgentState = {
            "user_text":      user_text,
            "steps":          [],
            "step_index":     0,
            "step_retries":   0,
            "task_retries":   0,
            "failed_steps":   0,
            "step_response":  "",
            "step_succeeded": False,
            "sentences":      [],
            "shared_messages": [],
            "interrupted":    False,
            "full_response":  "",
        }

        try:
            self._graph.invoke(initial_state)
        except Exception as e:
            print(f"\n  {C_ERR}[Graph Error] {e}{C_RESET}")
        finally:
            self._is_processing = False

    # ─── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self._is_running = True
        try:
            self._audio_stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                blocksize=AUDIO_BLOCKSIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._audio_stream.start()
            print(f"  {C_SYS}Microphone active. Listening...{C_RESET}\n")

            while self._is_running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n\n{C_SYS}  Agent offline. Goodbye.{C_RESET}\n")
        except Exception as e:
            print(f"\n  {C_ERR}[Fatal] {e}{C_RESET}")
        finally:
            self.shutdown()

    def shutdown(self):
        self._is_running = False
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
        self._tts_queue.put(None)
        self.tts.stop()
        if self.mcp:
            try:
                self.mcp.close()
            except Exception:
                pass


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = NaiyraAgent()
    agent.run()
