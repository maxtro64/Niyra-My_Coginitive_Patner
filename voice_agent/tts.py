"""
Text-to-Speech Engine
Supports:
1. Kokoro-82M (Local, default): 82M PyTorch model on CPU, custom voice files (.pt) supported.
2. Edge-TTS (Fallback): Microsoft Neural Voice — 0 RAM, needs internet.

The engine exposes `is_speaking` (true while playing + a short cooldown) so the microphone
listener can ignore the agent's own voice and avoid feedback loops.
"""

import asyncio
import concurrent.futures
import contextlib
import io
import os
import re
import threading
import time

import numpy as np
import sounddevice as sd

from core.config import (
    TTS_ECHO_COOLDOWN_S,
    TTS_EDGE_VOICE,
    TTS_ENGINE,
    TTS_SAMPLE_RATE,
    TTS_SPEED,
    TTS_VOICE,
)

_URL_RE = re.compile(r"https?://\S+")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_RE = re.compile(r"[*_#`>~|\[\]]+")


def clean_for_speech(text: str) -> str:
    """Strip markdown code blocks, URLs, and formatting so the voice sounds natural."""
    text = _CODE_BLOCK_RE.sub(" code omitted ", text or "")
    text = _URL_RE.sub(" link ", text)
    text = _MD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _run_coro(coro_factory):
    """Run an async factory to completion from sync code, even if this thread already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


class TextToSpeech:
    """High-quality text-to-speech with playback."""

    def __init__(self):
        self.engine = TTS_ENGINE.lower()
        self._voice = TTS_VOICE
        self._speed = TTS_SPEED
        self._sample_rate = TTS_SAMPLE_RATE
        self.pipeline = None

        self._speaking = threading.Event()
        self._last_end = 0.0

        if self.engine == "kokoro":
            print(f"  [TTS] Loading Kokoro TTS (voice={os.path.basename(str(TTS_VOICE))})...")
            try:
                from kokoro import KPipeline

                # Suppress Kokoro/PyTorch state_dict warnings during load
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    self.pipeline = KPipeline(lang_code="a")

                print("  [TTS] Kokoro loaded successfully.")
            except Exception as e:
                print(f"  [TTS] Kokoro failed ({e}), using Edge-TTS.")
                self.engine = "edge"
                self._voice = TTS_EDGE_VOICE
        else:
            self._voice = self._voice if "Neural" in str(self._voice) else TTS_EDGE_VOICE
            print(f"  [TTS] Edge-TTS initialized (voice={self._voice}, 0-RAM mode).")

    # ─── Echo gate support ────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        """True while audio plays and for a short cooldown afterwards (room reverb / speaker tail)."""
        return self._speaking.is_set() or (time.monotonic() - self._last_end) < TTS_ECHO_COOLDOWN_S

    # ─── Public API ───────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Synthesize text to speech and play it (blocking)."""
        text = clean_for_speech(text)
        if not text:
            return

        start_time = time.perf_counter()
        self._speaking.set()
        try:
            if self.engine == "kokoro" and self.pipeline is not None:
                if not self._speak_kokoro(text, start_time):
                    self._speak_edge_tts(text, start_time)
            else:
                self._speak_edge_tts(text, start_time)
        finally:
            self._last_end = time.monotonic()
            self._speaking.clear()

    def _speak_kokoro(self, text: str, start_time: float) -> bool:
        """Synthesize and play using Kokoro pipeline."""
        try:
            import torch
            with torch.inference_mode():
                audio_segments = []
                for gs, ps, audio in self.pipeline( # type: ignore
                    text,
                    voice=self._voice,
                    speed=self._speed
                ):
                    if audio is not None:
                        audio_np = audio.numpy() if hasattr(audio, 'numpy') else np.array(audio) # type: ignore
                        audio_segments.append(audio_np)

                if audio_segments:
                    full_audio = np.concatenate(audio_segments)
                    max_val = np.abs(full_audio).max()
                    if max_val > 0:
                        full_audio = full_audio / max_val * 0.9

                    sd.play(full_audio, samplerate=self._sample_rate)
                    sd.wait()
                    elapsed = (time.perf_counter() - start_time) * 1000
                    duration = len(full_audio) / self._sample_rate
                    print(f"  [TTS] Played {duration:.1f}s audio ({elapsed:.0f}ms total)")
                    return True
                return False
        except Exception as e:
            print(f"  [TTS] Kokoro synthesis error: {e}. Falling back to Edge-TTS...")
            return False

    def _speak_edge_tts(self, text: str, start_time: float):
        """Neural speech using Edge-TTS (needs internet)."""
        try:
            import edge_tts

            voice = self._voice if "Neural" in str(self._voice) else TTS_EDGE_VOICE

            async def _synthesize():
                communicate = edge_tts.Communicate(text, voice, pitch="-10Hz")
                chunks = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk.get("data", b"")) # type: ignore
                return b"".join(chunks)

            raw_audio = _run_coro(_synthesize)

            if raw_audio:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(io.BytesIO(raw_audio), format="mp3")
                sample_rate = audio_segment.frame_rate
                audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / 32768.0
                if audio_segment.channels > 1:
                    audio_data = audio_data.reshape((-1, audio_segment.channels))
                
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()
                elapsed = (time.perf_counter() - start_time) * 1000
                duration = len(audio_data) / sample_rate
                print(f"  [TTS] Spoken {duration:.1f}s via Edge-TTS ({elapsed:.0f}ms total)")
        except Exception as e:
            print(f"  [TTS] Edge-TTS error: {e}")

    def stop(self):
        """Stop any currently playing audio (for barge-in / abort)."""
        try:
            sd.stop()
        except Exception:
            pass

    def wait(self):
        """Wait for current playback to finish."""
        try:
            sd.wait()
        except Exception:
            pass
