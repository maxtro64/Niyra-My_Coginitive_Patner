"""
Text-to-Speech Engine
Supports:
1. Edge-TTS (Default): Ultra-realistic Microsoft Neural Voice (Ava/Emma) — 0 RAM, instant streaming, 10/10 human realism.
2. Kokoro-82M (Local): 82M PyTorch model on CPU when offline.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import io
import asyncio
from config import TTS_ENGINE, TTS_VOICE, TTS_SPEED, TTS_SAMPLE_RATE


class TextToSpeech:
    """High-quality text-to-speech with streaming playback."""

    def __init__(self):
        self.engine = TTS_ENGINE.lower()
        self._voice = TTS_VOICE
        self._speed = TTS_SPEED
        self._sample_rate = TTS_SAMPLE_RATE
        self.pipeline = None

        if self.engine == "kokoro":
            print(f"  [TTS] Loading Kokoro TTS (voice={TTS_VOICE})...")
            try:
                from kokoro import KPipeline
                self.pipeline = KPipeline(lang_code="a")
                print(f"  [TTS] Kokoro loaded successfully.")
            except Exception as e:
                print(f"  [TTS] Kokoro failed ({e}), using Edge-TTS.")
                self.engine = "edge"
                self._voice = "en-US-AvaNeural"
        else:
            print(f"  [TTS] Edge-TTS initialized (voice={self._voice}, 0-RAM mode).")

    def speak(self, text: str):
        """Synthesize text to speech and play it immediately."""
        if not text or not text.strip():
            return

        start_time = time.perf_counter()

        if self.engine == "kokoro" and self.pipeline is not None:
            success = self._speak_kokoro(text, start_time)
            if not success:
                self._speak_edge_tts(text, start_time)
        else:
            self._speak_edge_tts(text, start_time)

    def _speak_kokoro(self, text: str, start_time: float) -> bool:
        """Synthesize and play using Kokoro pipeline."""
        try:
            import torch
            with torch.inference_mode():
                audio_segments = []
                for gs, ps, audio in self.pipeline(
                    text,
                    voice=self._voice,
                    speed=self._speed
                ):
                    if audio is not None:
                        audio_np = audio.numpy() if hasattr(audio, 'numpy') else np.array(audio)
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
        """Ultra-realistic, 0-RAM neural speech using Edge-TTS."""
        try:
            import edge_tts

            voice = self._voice if "Neural" in self._voice else "en-US-AvaNeural"

            async def _synthesize():
                communicate = edge_tts.Communicate(text, voice)
                chunks = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
                return b"".join(chunks)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        raw_audio = pool.submit(asyncio.run, _synthesize()).result()
                else:
                    raw_audio = loop.run_until_complete(_synthesize())
            except RuntimeError:
                raw_audio = asyncio.run(_synthesize())

            if raw_audio:
                audio_data, sample_rate = sf.read(io.BytesIO(raw_audio))
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()
                elapsed = (time.perf_counter() - start_time) * 1000
                duration = len(audio_data) / sample_rate
                print(f"  [TTS] Spoken {duration:.1f}s via Edge-TTS ({elapsed:.0f}ms total)")
        except Exception as e:
            print(f"  [TTS] Edge-TTS error: {e}")

    def speak_streaming(self, text: str) -> float:
        """Synthesize and start playback, returning audio duration."""
        if not text or not text.strip():
            return 0.0

        if self.engine == "kokoro" and self.pipeline is not None:
            try:
                audio_segments = []
                for gs, ps, audio in self.pipeline(
                    text,
                    voice=self._voice,
                    speed=self._speed
                ):
                    if audio is not None:
                        audio_np = audio.numpy() if hasattr(audio, 'numpy') else np.array(audio)
                        audio_segments.append(audio_np)

                if audio_segments:
                    full_audio = np.concatenate(audio_segments)
                    max_val = np.abs(full_audio).max()
                    if max_val > 0:
                        full_audio = full_audio / max_val * 0.9

                    sd.play(full_audio, samplerate=self._sample_rate)
                    return len(full_audio) / self._sample_rate
            except Exception:
                pass

        # Speak via edge-tts
        self.speak(text)
        return 1.5

    def stop(self):
        """Stop any currently playing audio (for barge-in)."""
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
