"""
Bruce — Batcomputer Voice & Tactical Agent (Hermes Agent Style TUI)
Wayne Enterprises Defense Network // Gotham City Terminal

Usage:
    python bruce.py
"""

import os
import sys

# Ensure execution in the project virtual environment where all dependencies are installed.
# Cross-platform, and guarded by an env var so it can never relaunch in a loop.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if os.environ.get("NAIYRA_RELAUNCHED") != "1":
    for _candidate in (
        os.path.join(_root_dir, "venv", "Scripts", "python.exe"),
        os.path.join(_root_dir, "venv", "bin", "python"),
        os.path.join(_root_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(_root_dir, ".venv", "bin", "python"),
        os.path.join(_script_dir, "venv", "Scripts", "python.exe"),
        os.path.join(_script_dir, "venv", "bin", "python"),
    ):
        if os.path.exists(_candidate):
            if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(_candidate)):
                import subprocess
                print(f"[*] Re-launching under project venv: {_candidate}")
                os.environ["NAIYRA_RELAUNCHED"] = "1"
                sys.exit(subprocess.call([_candidate] + sys.argv))
            break

# Ensure both the root directory and voice_agent directory are in the path
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import asyncio
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd
try:
    from voice_agent.audio_io import MicListener
    from voice_agent.stt import SpeechToText
    from voice_agent.tts import TextToSpeech, clean_for_speech
    from voice_agent.vad import VoiceActivityDetector
except ImportError:
    from audio_io import MicListener
    from stt import SpeechToText
    from tts import TextToSpeech, clean_for_speech
    from vad import VoiceActivityDetector

from core.config import AUDIO_SAMPLE_RATE, BARGE_IN_ENABLED
from gui.main import BatcomputerGUI
from core.llm import MAX_STEP_RETRIES, MAX_TASK_RETRIES, LanguageModel, pop_sentences
from core.tools import SyncPlaywrightMcp
from core.supervisor_agent import SupervisorAgent

# Interrupt words — whole words only ("bruce" is an interrupt only while a task is running)
ABORT_RE = re.compile(r"\b(bruce|stop|cancel|abort)\b", re.IGNORECASE)


def check_and_unmute_windows_mic() -> tuple[bool, str]:
    """
    Check Windows Core Audio via pycaw.
    If the default microphone is muted, automatically unmute it and maximize volume.
    Returns (is_active, status_description). A failure to inspect the mic is reported
    as UNVERIFIED rather than as a healthy microphone.
    """
    if sys.platform != "win32":
        return True, "Mic status not checked (non-Windows)"
    try:
        try:
            import comtypes # type: ignore
            comtypes.CoInitialize()      # button handlers run on worker threads
        except Exception:
            pass
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore
        mic = AudioUtilities.GetMicrophone()
        if not mic:
            return False, "No Windows mic found"
        interface = mic.Activate(IAudioEndpointVolume._iid_, 23, None)
        vol = interface.QueryInterface(IAudioEndpointVolume)

        is_muted = vol.GetMute()
        if is_muted:
            vol.SetMute(0, None)
            is_muted = vol.GetMute()

        scalar = vol.GetMasterVolumeLevelScalar()
        if scalar < 0.5:
            vol.SetMasterVolumeLevelScalar(1.0, None)
            scalar = 1.0

        pct = int(scalar * 100)
        status_text = f"Default Mic ({pct}% Vol, {'MUTED' if is_muted else 'UNMUTED'})"
        return not is_muted, status_text
    except Exception as e:
        return True, f"Audio UNVERIFIED ({e})"


class BruceAgent:
    """
    Bruce — The Dark Knight Tactical Agent.
    Dual-mode: Real-time microphone listening + Hermes-style interactive terminal shell.

    Routing architecture:
      1. Every user utterance is first routed by the SupervisorAgent's LLM Router.
      2. If routed to 'general_agent' → falls back to the local streaming LanguageModel
         for real-time sentence-by-sentence TTS (conversational path).
      3. If routed to a specialized agent (web_agent, code_agent, etc.) → the Supervisor
         runs the task asynchronously and the result is spoken back via TTS.
    """

    def __init__(self, gui: BatcomputerGUI):
        self.gui = gui
        gui.log_system("BATCOMPUTER ONLINE")

        mic_ok, mic_desc = check_and_unmute_windows_mic()
        gui.set_mic_active(mic_ok, mic_desc)
        if not mic_ok:
            gui.log_error(f"Microphone unavailable: {mic_desc}")

        self.stt = SpeechToText()
        self.vad = VoiceActivityDetector()
        self.llm = LanguageModel()
        self.tts = TextToSpeech()
        gui.log_system("STT / VAD / LLM / TTS — ALL SYSTEMS ONLINE")

        # ── Supervisor Agent (multi-agent router) ────────────────────────────
        self.supervisor = None
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True
        )
        self._async_thread.start()

        try:
            self.supervisor = SupervisorAgent()
            gui.log_system("SUPERVISOR AGENT — Multi-agent router online")
        except Exception as e:
            gui.log_error(f"Supervisor init failed (fallback to direct LLM): {e}")
            self.supervisor = None

        # ── MCP Browser Tools ────────────────────────────────────────────────
        self.mcp = SyncPlaywrightMcp()
        def _connect_mcp_bg():
            try:
                self.mcp.connect(timeout=25)
                tools = self.mcp.get_tools()
                tool_names = [
                    (t.get("function", {}).get("name", "unknown") if isinstance(t, dict) else str(t))
                    for t in tools
                ]
                self.gui.set_tools(tool_names)
                self.gui.log_system(f"MCP TACTICAL TOOLS — {len(tools)} browser tools loaded")
            except Exception as e:
                self.gui.log_system(f"MCP background note: {e}")
        threading.Thread(target=_connect_mcp_bg, daemon=True).start()

        self._is_running = False
        self._is_processing = False
        self._interrupted = False
        self._mic_muted = False
        self._current_future = None

        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # Shared microphone listener (fixes audio queue overflow)
        self._listener = MicListener(
            vad=self.vad,
            on_speech=self._handle_transcription,
            is_muted=lambda: (not BARGE_IN_ENABLED and self.tts.is_speaking) or self._mic_muted,
            on_speech_start=lambda: self.gui.set_status("LISTENING"),
            on_error=lambda msg: self.gui.log_error(f"Audio error: {msg}"),
        )

        gui.set_input_callback(self._handle_user_text_input)
        gui.set_abort_callback(self._handle_abort)
        gui.set_testmic_callback(self._handle_testmic)
        gui.set_unmute_callback(self._handle_unmute)

        gui.log_system("AWAITING ORDERS. Speak or type.")
        gui.set_status("LISTENING")

    def _tts_worker(self):
        while True:
            try:
                sentence = self._tts_queue.get()
                if sentence is None:
                    break
                if not self._interrupted and sentence.strip():
                    self.tts.speak(sentence)
            except Exception as e:
                self.gui.log_error(f"TTS audio playback error: {e}")
            finally:
                self._tts_queue.task_done()

    def _handle_unmute(self):
        self._mic_muted = not self._mic_muted
        if self._mic_muted:
            self.gui.set_mic_active(False, "Default Mic (MUTED by operator)")
            self.gui.log_system("Microphone sensor MUTED. Type queries into prompt below.")
            self.gui.set_status("MIC MUTED")
        else:
            mic_ok, mic_desc = check_and_unmute_windows_mic()
            self.gui.set_mic_active(mic_ok, mic_desc)
            self.gui.log_system(f"Windows Microphone unmuted & active: {mic_desc}")
            self.gui.set_status("LISTENING")

    def _handle_testmic(self):
        def _test():
            self.gui.log_system("MIC DIAGNOSTIC — Speak now for 1.5 seconds...")
            try:
                rec = sd.rec(int(AUDIO_SAMPLE_RATE * 1.5), samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                max_val = float(np.max(np.abs(rec)))
                rms = float(np.sqrt(np.mean(rec ** 2)))
                mid = len(rec) // 2
                chunk = rec[mid:mid + 512, 0]
                if len(chunk) < 512:
                    chunk = np.pad(chunk, (0, 512 - len(chunk)))
                res = self.vad.process_chunk(chunk)
                conf = res["confidence"]
                if max_val < 0.0001:
                    self.gui.log_error("MIC DIAGNOSTIC: Near-zero signal. Check mic mute key (Fn+F4).")
                elif conf >= 0.35:
                    self.gui.log_system(f"MIC DIAGNOSTIC: Signal OK | Peak={max_val:.3f} | RMS={rms:.3f} | VAD={conf:.2f} — VOICE DETECTED")
                else:
                    self.gui.log_system(f"MIC DIAGNOSTIC: Signal OK | Peak={max_val:.3f} | RMS={rms:.3f} | VAD={conf:.2f} — Ambient/Quiet")
            except Exception as e:
                self.gui.log_error(f"Mic diagnostic failed: {e}")
        threading.Thread(target=_test, daemon=True).start()

    def _handle_user_text_input(self, text: str):
        if not text or not text.strip():
            return
        if self._is_processing:
            self.gui.log_system("Agent is busy processing. Type '/abort' or say 'Bruce' to interrupt.")
            return
        self.gui.log_user(text, is_voice=False)
        threading.Thread(target=self._process_utterance, args=(text,), daemon=True).start()

    def _handle_abort(self):
        self._interrupted = True
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except Exception:
                break
        self.tts.stop()
        if hasattr(self, "_current_future") and self._current_future is not None:
            try:
                self._current_future.cancel()
            except Exception:
                pass
        self._is_processing = False
        self.gui.stream_bruce_end()
        self.gui.log_system("OPERATOR ABORT SIGNAL RECEIVED — Tasks and speech terminated.")
        self.gui.set_status("LISTENING" if not self._mic_muted else "MIC MUTED")

    def _handle_transcription(self, audio: np.ndarray):
        user_text = self.stt.transcribe(audio)
        if not user_text or len(user_text.strip()) < 2:
            return

        if self._is_processing:
            if ABORT_RE.search(user_text):
                self.gui.log_system(f"INTERRUPTED — Heard: \"{user_text}\"")
                self._handle_abort()
            return

        self.gui.log_user(user_text, is_voice=True)
        self._process_utterance(user_text)

    def _run_async_loop(self):
        """Dedicated thread running the asyncio event loop for Supervisor tasks."""
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _run_supervisor_task(self, user_text: str) -> str | None:
        """Run a supervisor task on the async loop from a sync thread. Returns result string."""
        self._current_future = asyncio.run_coroutine_threadsafe(
            self.supervisor.run_task(user_text), self._async_loop
        )
        try:
            return self._current_future.result(timeout=300)  # 5 minute timeout
        except Exception as e:
            if not self._interrupted:
                self.gui.log_error(f"Supervisor task error: {e}")
            return None
        finally:
            self._current_future = None

    def _process_utterance(self, user_text: str):
        self._is_processing = True
        self._interrupted = False
        self.gui.set_status("THINKING")

        # ── Step 1: Route through SupervisorAgent ────────────────────────────
        routed_agent = None
        if self.supervisor:
            try:
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=user_text)]
                decision = self.supervisor.router.route(messages)
                routed_agent = decision.next_agent
                self.gui.log_system(
                    f"ROUTING → {routed_agent} (confidence: {decision.confidence:.2f}) — {decision.reasoning}"
                )
            except Exception as e:
                self.gui.log_error(f"Router error (fallback to direct LLM): {e}")
                routed_agent = "general_agent"
        else:
            routed_agent = "general_agent"

        # ── Step 2: Dispatch based on routing decision ───────────────────────
        if routed_agent in ("general_agent", "FINISH"):
            # Conversational / simple tasks → stream directly via local LLM + TTS
            self._process_via_streaming_llm(user_text)
        else:
            # Specialized agents → run full task via Supervisor, then speak result
            self._process_via_supervisor(user_text, routed_agent)

    def _process_via_streaming_llm(self, user_text: str):
        """Conversational path: stream through local LanguageModel for real-time TTS."""
        def on_tool_call(name, args): self.gui.log_tool(f"TOOL ▸ {name}")
        def on_tool_result(name, length): self.gui.log_tool(f"TOOL ◂ {name} ({length} chars)")
        def on_plan(steps):
            if not steps: return
            self.gui.log_system(f"MISSION PLAN — {len(steps)} steps queued")
            for i, s in enumerate(steps, 1): self.gui.log_system(f"  [{i}] {s}")
        def on_step(idx, step):
            self.gui.set_status(f"STEP {idx + 1}")
            self.gui.log_system(f"EXECUTING [{idx + 1}] {step}")
        def on_step_done(idx, ok):
            if ok: self.gui.log_system(f"COMPLETE  [{idx + 1}] ✓")
            else: self.gui.log_error(f"FAILED    [{idx + 1}] ✗ — step could not be verified")
        def on_step_retry(idx, att, step): self.gui.log_system(f"RETRY     [{idx + 1}] attempt {att}/{MAX_STEP_RETRIES} — {step[:60]}")
        def on_step_failed(idx, step): self.gui.log_error(f"STEP FAILED [{idx + 1}] max retries exhausted")
        def on_task_retry(att): self.gui.log_system(f"MISSION RESTART — re-planning (attempt {att}/{MAX_TASK_RETRIES})")

        try:
            self.gui.stream_bruce_start()
            for sentence in self.llm.generate_streaming(
                user_text,
                self.mcp,
                interrupt_check=lambda: self._interrupted,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_plan=on_plan,
                on_step=on_step,
                on_step_done=on_step_done,
                on_step_retry=on_step_retry,
                on_step_failed=on_step_failed,
                on_task_retry=on_task_retry,
            ):
                if self._interrupted:
                    break
                if not sentence.strip():
                    continue

                self.gui.stream_bruce_sentence(sentence)
                self.gui.set_status("SPEAKING")
                self._tts_queue.put(sentence)

            self.gui.stream_bruce_end()
            self._tts_queue.join()
            self.tts.wait()
            if self._interrupted:
                self.gui.log_system("MISSION ABORTED — Operator override.")

        except Exception as e:
            self.gui.stream_bruce_end()
            self.gui.log_error(str(e))

        finally:
            self._is_processing = False
            self.gui.set_status("LISTENING" if not self._mic_muted else "MIC MUTED")

    def _process_via_supervisor(self, user_text: str, agent_name: str):
        """Specialized agent path: Supervisor runs the full task, result is spoken back."""
        try:
            self.gui.log_system(f"DISPATCHING TO {agent_name.upper()}...")
            self.gui.set_status(f"AGENT: {agent_name.upper()}")

            result = self._run_supervisor_task(user_text)

            if self._interrupted:
                self.gui.log_system("MISSION ABORTED — Operator override.")
                return

            if result:
                # 1. Output tactical response in GUI log
                self.gui.stream_bruce_start()
                self.gui.set_status("SPEAKING")
                buffer = str(result)
                for line in buffer.split("\n"):
                    if line.strip():
                        self.gui.stream_bruce_sentence(line)
                self.gui.stream_bruce_end()

                # 2. Synthesize and speak final output through speakers
                speech_text = clean_for_speech(buffer)
                sentences, remainder = pop_sentences(speech_text)
                if not sentences and speech_text:
                    sentences = [speech_text]
                elif remainder and remainder.strip():
                    sentences.append(remainder.strip())

                for sentence in sentences:
                    if self._interrupted:
                        break
                    if sentence.strip():
                        self._tts_queue.put(sentence)

                # Wait for audio synthesis and playback to fully complete
                self._tts_queue.join()
                self.tts.wait()
            else:
                self.gui.log_error(f"{agent_name} returned no result.")

        except Exception as e:
            self.gui.log_error(f"Supervisor execution error: {e}")

        finally:
            self._is_processing = False
            self.gui.set_status("LISTENING" if not self._mic_muted else "MIC MUTED")

    def run(self):
        self._is_running = True
        try:
            self._listener.start()
            self.gui.log_system("Acoustic Sensor stream active. Listening...")
            while self._is_running:
                time.sleep(0.1)
        except Exception as e:
            self.gui.log_error(f"Microphone init error: {e}. You can still type queries at the prompt below!")
            self.gui.set_mic_active(False, f"Error: {e}")
            while self._is_running:
                time.sleep(0.1)
        finally:
            self.shutdown()

    def shutdown(self):
        self._is_running = False
        self._interrupted = True
        if hasattr(self, "_listener"):
            self._listener.stop()
        self._tts_queue.put(None)
        self.tts.stop()
        if hasattr(self, 'mcp') and self.mcp is not None:
            try:
                self.mcp.close()
            except Exception:
                pass
        # Shut down the async event loop
        if hasattr(self, '_async_loop') and self._async_loop.is_running():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    gui = BatcomputerGUI()
    agent = None
    try:
        agent = BruceAgent(gui)
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()
        gui.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        gui.log_error(f"Fatal startup error: {e}")
    finally:
        if agent is not None:
            agent.shutdown()
