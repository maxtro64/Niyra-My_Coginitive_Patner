"""
Naiyra Voice Agent — Main Pipeline
VAD → STT → LLM → TTS with sentence-level streaming.

Usage:
    python agent.py
"""

import numpy as np
import sounddevice as sd
import threading
import time
import sys

from config import AUDIO_SAMPLE_RATE, AUDIO_BLOCKSIZE, AUDIO_CHANNELS
from vad import VoiceActivityDetector
from stt import SpeechToText
from llm import LanguageModel
from tts import TextToSpeech

# ─── ANSI Colors for terminal output ──────────────────────────────────
try:
    from colorama import init, Fore, Style
    init()
    C_USER = Fore.CYAN
    C_NAIYRA = Fore.MAGENTA
    C_SYSTEM = Fore.YELLOW
    C_DIM = Style.DIM
    C_RESET = Style.RESET_ALL
    C_BOLD = Style.BRIGHT
except ImportError:
    C_USER = C_NAIYRA = C_SYSTEM = C_DIM = C_RESET = C_BOLD = ""


class NaiyraAgent:
    """
    Real-time voice agent with streaming pipeline:
    Microphone → Silero VAD → Faster-Whisper STT → Phi3 LLM → Kokoro TTS → Speaker
    """

    def __init__(self):
        print(f"\n{C_SYSTEM}{'='*60}")
        print(f"  🎙️  NAIYRA — Voice Agent Initializing...")
        print(f"{'='*60}{C_RESET}\n")

        # Load all components (STT first to ensure MKL pool initializes before PyTorch memory expansion)
        self.stt = SpeechToText()
        self.vad = VoiceActivityDetector()
        self.llm = LanguageModel()
        self.tts = TextToSpeech()

        # State
        self._is_running = False
        self._is_processing = False  # True while LLM+TTS is generating
        self._audio_stream = None
        self._speech_audio_chunks = []  # Buffer for current speech

        print(f"\n{C_SYSTEM}{'='*60}")
        print(f"  ✅ All systems loaded. Ready to chat!")
        print(f"  📢 Speak into your microphone to begin.")
        print(f"  ⌨️  Press Ctrl+C to quit.")
        print(f"{'='*60}{C_RESET}\n")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice for every audio chunk from the microphone.
        Runs in a separate thread — must be fast.
        """
        if status:
            pass  # Ignore overflow/underflow for now

        # Convert to float32 mono
        audio_chunk = indata[:, 0].copy().astype(np.float32)

        # If we're currently playing a response, skip VAD processing
        # (prevents echo/feedback loop)
        if self._is_processing:
            return

        # Process through VAD in 512-sample chunks
        chunk_size = 512
        for i in range(0, len(audio_chunk), chunk_size):
            sub_chunk = audio_chunk[i:i + chunk_size]
            if len(sub_chunk) < chunk_size:
                # Pad last chunk
                sub_chunk = np.pad(sub_chunk, (0, chunk_size - len(sub_chunk)))

            result = self.vad.process_chunk(sub_chunk)

            if result["event"] == "speech_start":
                self._speech_audio_chunks = []
                print(f"\n{C_DIM}  🎤 Listening...{C_RESET}", end="", flush=True)

            if self.vad.is_speaking or result["event"] == "speech_end":
                self._speech_audio_chunks.append(sub_chunk)

            if result["event"] == "speech_end":
                print(f"\r{' '*40}\r", end="", flush=True)  # Clear "Listening..."
                # Collect speech and process in background thread
                speech_audio = np.concatenate(self._speech_audio_chunks)
                self._speech_audio_chunks = []
                self.vad.reset()

                # Process in background to not block audio callback
                threading.Thread(
                    target=self._process_utterance,
                    args=(speech_audio,),
                    daemon=True,
                ).start()

    def _process_utterance(self, audio: np.ndarray):
        """
        Full pipeline: STT → LLM → TTS for a captured utterance.
        Runs in a background thread.
        """
        self._is_processing = True

        try:
            # ── Step 1: Transcribe ──
            user_text = self.stt.transcribe(audio)

            if not user_text or len(user_text.strip()) < 2:
                self._is_processing = False
                return

            print(f"  {C_BOLD}{C_USER}You:{C_RESET} {user_text}")

            # ── Step 2 + 3: LLM → TTS with sentence streaming ──
            print(f"  {C_BOLD}{C_NAIYRA}Naiyra:{C_RESET} ", end="", flush=True)

            full_response = ""
            for sentence in self.llm.generate_streaming(user_text):
                if not sentence.strip():
                    continue

                full_response += sentence + " "
                print(f"{sentence} ", end="", flush=True)

                # Speak this sentence immediately
                self.tts.speak(sentence)

            print()  # Newline after full response

        except Exception as e:
            print(f"\n  {C_SYSTEM}[Error] {e}{C_RESET}")

        finally:
            self._is_processing = False

    def run(self):
        """Start the voice agent loop."""
        self._is_running = True

        try:
            # Open microphone stream
            self._audio_stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                blocksize=AUDIO_BLOCKSIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._audio_stream.start()
            print(f"  {C_SYSTEM}🎧 Microphone active. Start talking!{C_RESET}\n")

            # Keep main thread alive
            while self._is_running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n\n{C_SYSTEM}  👋 Naiyra signing off. Goodbye!{C_RESET}\n")
        except Exception as e:
            print(f"\n  {C_SYSTEM}[Fatal Error] {e}{C_RESET}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of all components."""
        self._is_running = False
        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
        self.tts.stop()


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = NaiyraAgent()
    agent.run()
