"""
Bruce — Batcomputer Voice & Tactical Agent (Hermes Agent Style TUI)
Wayne Enterprises Defense Network // Gotham City Terminal

Usage:
    python bruce.py
"""

import os
import sys

# Ensure execution in project virtual environment where all dependencies are installed
_script_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_script_dir, "venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.normpath(sys.executable).lower() != os.path.normpath(_venv_python).lower():
    import subprocess
    print(f"[*] Re-launching under project venv: {_venv_python}")
    sys.exit(subprocess.call([_venv_python] + sys.argv))

import queue
import time
import threading
import numpy as np
import sounddevice as sd

from config import AUDIO_SAMPLE_RATE, AUDIO_BLOCKSIZE, AUDIO_CHANNELS
from vad import VoiceActivityDetector
from stt import SpeechToText
from llm import LanguageModel
from tts import TextToSpeech
from tools import SyncPlaywrightMcp
from gui import BatcomputerGUI


def check_and_unmute_windows_mic() -> tuple[bool, str]:
    """
    Check Windows Core Audio via pycaw.
    If the default microphone is muted, automatically unmute it and maximize volume.
    Returns (is_active, status_description).
    """
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
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
        return True, f"Audio OK ({e})"


class BruceAgent:
    """
    Bruce — The Dark Knight Tactical Agent.
    Dual-mode: Real-time microphone listening + Hermes-style interactive terminal shell.
    """

    def __init__(self, gui: BatcomputerGUI):
        self.gui = gui

        gui.log_system("BATCOMPUTER ONLINE")

        # Check and auto-unmute Windows microphone
        mic_ok, mic_desc = check_and_unmute_windows_mic()
        gui.set_mic_active(mic_ok, mic_desc)
        if not mic_ok:
            gui.log_error(f"Microphone unavailable: {mic_desc}")

        # Load all components
        self.stt = SpeechToText()
        self.vad = VoiceActivityDetector()
        self.llm = LanguageModel()
        self.tts = TextToSpeech()

        gui.log_system("STT / VAD / LLM / TTS — ALL SYSTEMS ONLINE")

        self.mcp = SyncPlaywrightMcp()
        try:
            self.mcp.connect()
            tools = self.mcp.get_tools()
            tool_names = []
            for t in tools:
                name = t.get("function", {}).get("name", "unknown") if isinstance(t, dict) else str(t)
                tool_names.append(name)
            gui.set_tools(tool_names)
            gui.log_system(f"MCP TACTICAL TOOLS — {len(tools)} tools loaded")
        except Exception as e:
            gui.log_error(f"MCP connection failed: {e}")
            self.mcp = None

        # State & Threading
        self._is_running = False
        self._is_processing = False
        self._interrupted = False
        self._audio_stream = None

        # TTS Queue for concurrent speech synthesis
        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # Connect GUI Callbacks
        gui.set_input_callback(self._handle_user_text_input)
        gui.set_abort_callback(self._handle_abort)
        gui.set_testmic_callback(self._handle_testmic)
        gui.set_unmute_callback(self._handle_unmute)

        gui.log_system("AWAITING ORDERS. Speak or type.")
        gui.set_status("LISTENING")

    def _tts_worker(self):
        """Worker thread that consumes and speaks sentences sequentially."""
        while True:
            try:
                sentence = self._tts_queue.get()
                if sentence is None:  # Shutdown sentinel
                    break
                if not self._interrupted and sentence.strip():
                    self.tts.speak(sentence)
                self._tts_queue.task_done()
            except Exception:
                pass

    def _handle_unmute(self):
        """Force unmute Windows microphone."""
        mic_ok, mic_desc = check_and_unmute_windows_mic()
        self.gui.set_mic_active(mic_ok, mic_desc)
        self.gui.log_system(f"Windows Microphone unmuted: {mic_desc}")

    def _handle_testmic(self):
        """Record 1.5 seconds from microphone and display diagnostics."""
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
                    self.gui.log_system(f"MIC DIAGNOSTIC: Signal OK | Peak={max_val:.3f} | VAD={conf:.2f} — VOICE DETECTED")
                else:
                    self.gui.log_system(f"MIC DIAGNOSTIC: Signal OK | Peak={max_val:.3f} | VAD={conf:.2f} — Ambient/Quiet")
            except Exception as e:
                self.gui.log_error(f"Mic diagnostic failed: {e}")

        threading.Thread(target=_test, daemon=True).start()

    def _handle_user_text_input(self, text: str):
        """Handle command typed by operator in the terminal."""
        if not text or not text.strip():
            return
        if self._is_processing:
            self.gui.log_system("Agent is busy processing. Type '/abort' or say 'Bruce' to interrupt.")
            return
        self.gui.log_user(text, is_voice=False)
        threading.Thread(target=self._process_utterance, args=(text,), daemon=True).start()

    def _handle_abort(self):
        """Handle operator abort / stop request."""
        if self._is_processing:
            self._interrupted = True
            # Clear pending TTS sentences
            while not self._tts_queue.empty():
                try:
                    self._tts_queue.get_nowait()
                    self._tts_queue.task_done()
                except Exception:
                    break
            self.tts.stop()
            self.gui.log_system("OPERATOR ABORT SIGNAL RECEIVED — Tasks terminated.")
            self.gui.set_status("LISTENING")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for every audio chunk from the microphone."""
        audio_chunk = indata[:, 0].copy().astype(np.float32)

        # Process through VAD in 512-sample chunks
        chunk_size = 512
        for i in range(0, len(audio_chunk), chunk_size):
            sub_chunk = audio_chunk[i:i + chunk_size]
            if len(sub_chunk) < chunk_size:
                sub_chunk = np.pad(sub_chunk, (0, chunk_size - len(sub_chunk)))

            result = self.vad.process_chunk(sub_chunk)

            if result["event"] == "speech_start":
                self.gui.set_status("LISTENING")

            if result["event"] == "speech_end":
                speech_audio = self.vad.get_speech_audio()
                self.vad.reset()

                if len(speech_audio) > 1600:  # At least 100ms of audio
                    threading.Thread(
                        target=self._handle_transcription,
                        args=(speech_audio,),
                        daemon=True,
                    ).start()

    def _handle_transcription(self, audio: np.ndarray):
        """Transcribe and decide whether to interrupt or start new processing."""
        user_text = self.stt.transcribe(audio)

        if not user_text or len(user_text.strip()) < 2:
            return

        text_lower = user_text.lower()

        if self._is_processing:
            if "bruce" in text_lower or "stop" in text_lower or "cancel" in text_lower:
                self.gui.log_system(f"INTERRUPTED — Heard: \"{user_text}\"")
                self._handle_abort()
            return

        # Normal input from voice
        self.gui.log_user(user_text, is_voice=True)
        self._process_utterance(user_text)

    def _process_utterance(self, user_text: str):
        """Full pipeline: Plan → Execute Steps → Stream to TUI + TTS."""
        self._is_processing = True
        self._interrupted = False
        self.gui.set_status("THINKING")

        def on_tool_call(tool_name, tool_args):
            self.gui.log_tool(f"TOOL ▸ {tool_name}")

        def on_tool_result(tool_name, length):
            self.gui.log_tool(f"TOOL ◂ {tool_name} ({length} bytes)")

        def on_plan(steps):
            self.gui.log_system(f"MISSION PLAN — {len(steps)} steps queued")
            for i, s in enumerate(steps, 1):
                self.gui.log_system(f"  [{i}] {s}")

        def on_step(idx, step_text):
            self.gui.set_status(f"STEP {idx + 1}")
            self.gui.log_system(f"EXECUTING [{idx + 1}] {step_text}")

        def on_step_done(idx, succeeded):
            if succeeded:
                self.gui.log_system(f"COMPLETE  [{idx + 1}] ✓")
            else:
                self.gui.log_error(f"FAILED    [{idx + 1}] ✗ — step could not be verified")

        def on_step_retry(idx, attempt, step_text):
            self.gui.log_system(f"RETRY     [{idx + 1}] attempt {attempt}/{3} — {step_text[:60]}")

        def on_step_failed(idx, step_text):
            self.gui.log_error(f"STEP FAILED [{idx + 1}] max retries exhausted")

        def on_task_retry(attempt):
            self.gui.log_system(f"MISSION RESTART — re-planning (attempt {attempt}/2)")

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

                # Stream sentence immediately to the terminal GUI
                self.gui.stream_bruce_sentence(sentence)

                # Queue sentence for concurrent TTS speech
                self.gui.set_status("SPEAKING")
                self._tts_queue.put(sentence)

            self.gui.stream_bruce_end()

            if self._interrupted:
                self.gui.log_system("MISSION ABORTED — Operator override.")

        except Exception as e:
            self.gui.stream_bruce_end()
            self.gui.log_error(str(e))

        finally:
            self._is_processing = False
            self.gui.set_status("LISTENING")

    def run(self):
        """Start the voice agent microphone stream in background thread."""
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
            self.gui.log_system("Acoustic Sensor stream active. Listening...")
        except Exception as e:
            self.gui.log_error(f"Microphone init error: {e}. You can still type queries at the prompt below!")
            self.gui.set_mic_active(False, f"Error: {e}")

        try:
            while self._is_running:
                time.sleep(0.1)
        except Exception as e:
            self.gui.log_error(f"Agent runtime error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components."""
        self._is_running = False
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
        self._tts_queue.put(None)
        self.tts.stop()
        if hasattr(self, 'mcp') and self.mcp is not None:
            try:
                self.mcp.close()
            except Exception:
                pass


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create the Batcomputer Terminal Interface
    gui = BatcomputerGUI()

    try:
        # Initialize Bruce Agent subsystems (STT, VAD, LLM, TTS, MCP)
        agent = BruceAgent(gui)

        # Start microphone listening stream in background thread
        agent_thread = threading.Thread(target=agent.run, daemon=True)
        agent_thread.start()

        # Run the interactive Batcomputer terminal loop on the main thread
        gui.run()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        gui.log_error(f"Fatal startup error: {e}")
    finally:
        if 'agent' in locals():
            agent.shutdown()
