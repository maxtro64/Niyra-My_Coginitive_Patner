"""
Naiyra Voice Agent — LangGraph Execution Graph
VAD → STT → [Plan → Execute → Verify → Retry] → TTS

Graph nodes:
  plan         : LLM decomposes task into steps (or marks as direct)
  execute      : LLM executes current step with browser tools
  verify       : LLM + browser_snapshot verify step completion
  retry        : Increment retry counter, modify prompt
  advance      : Move to the next step
  mark_failed  : Step exhausted retries
  replan       : Task-level retry with a new plan
  finalize     : Update history, reset state

Edge conditions:
  after execute: interrupted → finalize | else verify
  after verify: SUCCESS → next step (or done) | FAILURE + retries left → retry | exhausted → mark_failed
  after mark_failed: too many failed steps → replan | else next step
  An interrupt (abort / voice "stop") short-circuits every edge to finalize.
"""

import operator
import queue
import re
import os
import sys
import threading
import time
from typing import Annotated, Any, TypedDict

import numpy as np

# Ensure root and voice_agent directories are on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from voice_agent.audio_io import MicListener
    from voice_agent.stt import SpeechToText
    from voice_agent.tts import TextToSpeech
    from voice_agent.vad import VoiceActivityDetector
except ImportError:
    from audio_io import MicListener
    from stt import SpeechToText
    from tts import TextToSpeech
    from vad import VoiceActivityDetector

from langgraph.graph import END, StateGraph

from core.config import BARGE_IN_ENABLED, GRAPH_RECURSION_LIMIT
from core.llm import MAX_STEP_RETRIES, MAX_TASK_RETRIES, LanguageModel
from core.tools import SyncPlaywrightMcp

# ─── ANSI Colors ─────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init
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

ABORT_RE = re.compile(r"\b(stop|cancel|abort)\b", re.IGNORECASE)






# ─── Graph State ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Input
    user_text: str                      # The current user utterance

    # Planning
    steps: list[str]                    # Decomposed step list (empty = direct reply)
    step_index: int                     # Which step we're on
    step_retries: int                   # Retry count for current step
    task_retries: int                   # Full task restart count
    failed_steps: int                   # Steps that exhausted retries

    # Execution
    step_response: str                  # LLM spoken output for current step
    step_succeeded: bool                # Did verifier confirm success?
    sentences: Annotated[list[str], operator.add]  # Accumulates spoken sentences

    # Context
    shared_messages: list[dict]         # Persistent LLM message thread (browser session)
    msgs_checkpoint: int                # Thread length before the current step (retry revert point)
    interrupted: bool                   # Abort flag

    # Output
    full_response: str                  # Complete response for history


# ─── Node Implementations ────────────────────────────────────────────────────

def node_plan(state: AgentState, llm: LanguageModel, mcp: Any) -> dict:
    """
    Plan node: decompose user_text into atomic steps.
    If it's a direct query, steps = [].
    """
    user_text = state["user_text"]

    steps: list[str] = []
    if llm._is_complex_task(user_text):
        planned = llm._plan_task(user_text, mcp)
        if planned and [s.lower() for s in planned] != ["respond directly"]:
            steps = planned
            print(f"\n{C_STEP}  ◈ MISSION PLAN — {len(steps)} steps{C_RESET}")
            for i, s in enumerate(steps, 1):
                print(f"{C_STEP}    [{i}] {s}{C_RESET}")

    plan_ctx = None
    if steps:
        plan_ctx = (
            f"The user has requested: {user_text}\n\n"
            f"Mission plan ({len(steps)} steps — execute in order):\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
            + "\n\nIMPORTANT: Browser state is preserved between steps."
        )

    return {
        "steps": steps,
        "step_index": 0,
        "step_retries": 0,
        "task_retries": state.get("task_retries", 0),
        "failed_steps": 0,
        "sentences": [],
        "shared_messages": llm.build_shared_messages(plan_ctx),
        "msgs_checkpoint": 0,
        "full_response": "",
        "step_response": "",
        "step_succeeded": False,
        "interrupted": False,
    }


def node_execute_step(state: AgentState, llm: LanguageModel, mcp: Any,
                      speak, interrupt_check) -> dict:
    """
    Execute node: run the current step (or full direct reply if no plan).
    Streams sentences to the TTS queue in real-time (never blocks on audio playback).
    Uses the SHARED message thread so browser session context is preserved.
    """
    steps       = state["steps"]
    idx         = state["step_index"]
    retries     = state["step_retries"]
    shared_msgs = state["shared_messages"]
    user_text   = state["user_text"]

    # Same gating function as llm.generate_streaming — no more drift between the two paths
    llm_tools = llm.get_llm_tools(mcp, user_text, force=bool(steps))

    # Retry: throw away the failed attempt's tool calls/results from the thread
    if retries == 0:
        checkpoint = len(shared_msgs)
    else:
        checkpoint = state.get("msgs_checkpoint", len(shared_msgs))
        del shared_msgs[checkpoint:]

    if not steps:
        prompt = user_text
        print(f"\n{C_AGENT}  ◈ RESPONDING DIRECTLY...{C_RESET}")
    else:
        step = steps[idx]
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
                f"Try differently: use browser_snapshot first, then the exact element ref from it. "
                f"Speak result in 1-2 sentences."
            )
            print(f"\n{C_STEP}  ↻ RETRY [{idx+1}] attempt {retries}/{MAX_STEP_RETRIES}{C_RESET}")

    sentences_this_step: list[str] = []
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
        speak(sentence)          # queued — generation and verification continue while audio plays

    print()  # newline after response

    return {
        "step_response": full_text.strip(),
        "sentences": sentences_this_step,
        "shared_messages": shared_msgs,
        "msgs_checkpoint": checkpoint,
        "full_response": state.get("full_response", "") + full_text,
        "interrupted": bool(interrupt_check()),
    }


def node_verify_step(state: AgentState, llm: LanguageModel, mcp: Any) -> dict:
    """
    Verify node: check whether the current step succeeded.
    Uses browser_snapshot + spoken output for judgement.
    """
    if state.get("interrupted"):
        return {"step_succeeded": False}

    steps  = state["steps"]
    idx    = state["step_index"]
    spoken = state["step_response"]

    if not steps:
        return {"step_succeeded": True}      # Direct response — always passes

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
        "step_index":     new_idx,
        "step_retries":   0,
        "step_succeeded": False,   # must reset: stale True would skip verify next round
        "step_response":  "",
    }


def node_retry_step(state: AgentState) -> dict:
    """Increment retry counter — clear stale success flag before re-executing."""
    return {
        "step_retries":   state["step_retries"] + 1,
        "step_succeeded": False,
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
        "step_succeeded": False,
        "step_response": "",
    }


def node_replan(state: AgentState, llm: LanguageModel, mcp: Any) -> dict:
    """
    Task retry: re-plan with failure context, restart step index.
    """
    attempt = state["task_retries"] + 1
    print(f"\n{C_ERR}  ↻ MISSION RESTART — re-planning (attempt {attempt}/{MAX_TASK_RETRIES}){C_RESET}")

    new_steps = llm._plan_task(
        f"{state['user_text']}\n\nPrevious attempt failed on multiple steps. "
        f"Re-plan with more specific, atomic steps using exact URLs and element descriptions.",
        mcp,
    ) or state["steps"]

    print(f"{C_STEP}  ◈ NEW PLAN — {len(new_steps)} steps{C_RESET}")
    for i, s in enumerate(new_steps, 1):
        print(f"{C_STEP}    [{i}] {s}{C_RESET}")

    plan_ctx = (
        f"The user has requested: {state['user_text']}\n\n"
        f"Revised mission plan ({len(new_steps)} steps):\n"
        + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(new_steps))
    )

    return {
        "steps": new_steps,
        "step_index": 0,
        "step_retries": 0,
        "failed_steps": 0,
        "task_retries": attempt,
        "step_succeeded": False,
        "step_response": "",
        "shared_messages": llm.build_shared_messages(plan_ctx),   # fresh thread for the new attempt
        "msgs_checkpoint": 0,
    }


def node_finalize(state: AgentState, llm: LanguageModel) -> dict:
    """Update conversation history, return final state."""
    llm.commit_history(state["user_text"], state.get("full_response", ""))

    steps_done = len(state.get("steps", []))
    failed     = state.get("failed_steps", 0)
    if steps_done and not state.get("interrupted"):
        print(f"\n{C_SYS}  ◈ Mission complete: {steps_done - failed}/{steps_done} steps verified.{C_RESET}")

    return {}


# ─── Edge Conditions ─────────────────────────────────────────────────────────

def edge_after_execute(state: AgentState) -> str:
    return "finalize" if state.get("interrupted") else "verify"


def edge_after_verify(state: AgentState) -> str:
    """Route after verification — SUCCESS goes straight to next step (no intermediate node)."""
    if state.get("interrupted"):
        return "finalize"

    steps   = state["steps"]
    idx     = state["step_index"]
    retries = state["step_retries"]

    if not steps:
        return "finalize"          # Direct reply — done

    if state["step_succeeded"]:
        if idx + 1 >= len(steps):
            return "finalize"
        return "advance"

    if retries < MAX_STEP_RETRIES:
        return "retry"
    return "mark_failed"


def edge_after_advance(state: AgentState) -> str:
    """After incrementing step_index, go execute or finalize."""
    if state.get("interrupted") or state["step_index"] >= len(state["steps"]):
        return "finalize"
    return "execute"


def edge_after_mark_failed(state: AgentState) -> str:
    """After marking a step failed, decide: continue or re-plan task."""
    if state.get("interrupted"):
        return "finalize"

    failed = state["failed_steps"]
    total  = len(state["steps"])
    task_retries = state["task_retries"]

    # If more than half failed and we still have task retries left → re-plan
    if failed >= max(1, total // 2) and task_retries < MAX_TASK_RETRIES:
        return "replan"

    if state["step_index"] >= total:
        return "finalize"
    return "execute"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_agent_graph(llm: LanguageModel, mcp: Any, speak, interrupt_check) -> Any:
    """
    Build and compile the LangGraph execution graph.

    `speak` is a callable that QUEUES a sentence for TTS (it must not block).
    """
    g = StateGraph(AgentState)

    g.add_node("plan",        lambda s: node_plan(s, llm, mcp)) # type: ignore
    g.add_node("execute",     lambda s: node_execute_step(s, llm, mcp, speak, interrupt_check)) # type: ignore
    g.add_node("verify",      lambda s: node_verify_step(s, llm, mcp)) # type: ignore
    g.add_node("advance",     lambda s: node_advance_step(s)) # type: ignore
    g.add_node("retry",       lambda s: node_retry_step(s)) # type: ignore
    g.add_node("mark_failed", lambda s: node_mark_step_failed(s)) # type: ignore
    g.add_node("replan",      lambda s: node_replan(s, llm, mcp)) # type: ignore
    g.add_node("finalize",    lambda s: node_finalize(s, llm)) # type: ignore

    g.set_entry_point("plan")

    g.add_edge("plan", "execute")

    g.add_conditional_edges("execute", edge_after_execute, {
        "verify":   "verify",
        "finalize": "finalize",
    })

    g.add_conditional_edges("verify", edge_after_verify, {
        "advance":     "advance",
        "retry":       "retry",
        "mark_failed": "mark_failed",
        "finalize":    "finalize",
    })

    g.add_conditional_edges("advance", edge_after_advance, {
        "execute":  "execute",
        "finalize": "finalize",
    })

    g.add_edge("retry", "execute")

    g.add_conditional_edges("mark_failed", edge_after_mark_failed, {
        "execute":  "execute",
        "replan":   "replan",
        "finalize": "finalize",
    })

    g.add_edge("replan", "execute")
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
        print("  NAIYRA — LangGraph Voice Agent Initializing...")
        print(f"{'='*60}{C_RESET}\n")

        # Init order matters on 8GB machines: CTranslate2 (STT) must claim MKL before PyTorch (VAD)
        self.stt = SpeechToText()
        self.vad = VoiceActivityDetector()
        self.llm = LanguageModel()
        self.tts = TextToSpeech()

        print(f"  {C_SYS}Connecting to MCP / Browser...{C_RESET}")
        self.mcp = SyncPlaywrightMcp()
        try:
            self.mcp.connect()
            tools = self.mcp.get_tools()
            print(f"  {C_SYS}Tools online — {len(tools)} loaded.{C_RESET}")
        except Exception as e:
            # Keep self.mcp: native tools (Tavily) still work even when Playwright failed
            print(f"  {C_ERR}MCP connection problem: {e}{C_RESET}")

        # State
        self._is_running    = False
        self._is_processing = False
        self._interrupted   = False
        self._state_lock    = threading.Lock()

        # TTS queue (speak sentences as they arrive; graph never waits for audio)
        self._tts_queue: queue.Queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # Microphone: VAD runs in a worker thread, muted while the agent speaks (unless barge-in)
        self._listener = MicListener(
            self.vad,
            on_speech=self._handle_speech,
            is_muted=self._mic_muted,
            on_error=lambda m: print(f"  {C_ERR}{m}{C_RESET}"),
        )

        self._graph = build_agent_graph(
            self.llm, self.mcp, self._enqueue_tts,
            interrupt_check=lambda: self._interrupted
        )

        print(f"\n{C_BOLD}{'='*60}")
        print("  ✓ All systems online. Speak or press Ctrl+C to quit.")
        print(f"{'='*60}{C_RESET}\n")

    # ─── TTS worker ───────────────────────────────────────────────────────────

    def _enqueue_tts(self, sentence: str):
        self._tts_queue.put(sentence)

    def _tts_worker(self):
        while True:
            item = self._tts_queue.get()
            try:
                if item is None:
                    break
                if not self._interrupted and item.strip():
                    self.tts.speak(item)
            except Exception as e:
                print(f"  {C_ERR}[TTS] {e}{C_RESET}")
            finally:
                self._tts_queue.task_done()

    def _mic_muted(self) -> bool:
        """Echo gate: ignore the microphone while the agent talks, unless barge-in is enabled."""
        return (not BARGE_IN_ENABLED) and (self.tts.is_speaking or not self._tts_queue.empty())

    # ─── Speech handling ──────────────────────────────────────────────────────

    def _try_begin(self) -> bool:
        """Atomically claim the 'processing' slot so two utterances never start two graphs."""
        with self._state_lock:
            if self._is_processing:
                return False
            self._is_processing = True
            self._interrupted = False
            return True

    def _abort(self):
        self._interrupted = True
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except Exception:
                break
        self.tts.stop()

    def _handle_speech(self, audio: np.ndarray):
        user_text = self.stt.transcribe(audio)
        if not user_text or len(user_text.strip()) < 2:
            return

        # While a task is running the mic is still live (it is only muted during playback),
        # so "stop" / "cancel" / "abort" work during planning and tool execution.
        if self._is_processing:
            if ABORT_RE.search(user_text):
                print(f"\n  {C_SYS}[Abort] heard: \"{user_text}\"{C_RESET}")
                self._abort()
            return

        if not self._try_begin():
            return

        print(f"\n  {C_BOLD}{C_USER}You:{C_RESET} {user_text}")
        self._run_graph(user_text)

    # ─── LangGraph invocation ─────────────────────────────────────────────────

    def _run_graph(self, user_text: str):
        """Invoke the LangGraph pipeline for a user utterance (caller already claimed the slot)."""
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
            "msgs_checkpoint": 0,
            "interrupted":    False,
            "full_response":  "",
        }

        try:
            # Each step is execute→verify→advance (3 visits); the default limit of 25 would
            # raise GraphRecursionError on a 9-12 step plan.
            self._graph.invoke(initial_state, {"recursion_limit": GRAPH_RECURSION_LIMIT})
        except Exception as e:
            print(f"\n  {C_ERR}[Graph Error] {e}{C_RESET}")
        finally:
            self._is_processing = False

    # ─── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self._is_running = True
        try:
            self._listener.start()
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
        self._interrupted = True
        self._listener.stop()
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
