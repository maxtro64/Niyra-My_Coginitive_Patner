"""
Faster-Whisper STT — Speech to Text
CTranslate2 implementation of OpenAI Whisper for fast, accurate transcription.
"""

import threading
import time

import numpy as np
from faster_whisper import WhisperModel

from core.config import (
    STT_BEAM_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_LANGUAGE,
    STT_MODEL_SIZE,
)


class SpeechToText:
    """Fast speech transcription using Faster-Whisper (CPU INT8 by default, see config)."""

    def __init__(self):
        print(f"  [STT] Loading Faster-Whisper '{STT_MODEL_SIZE}' on {STT_DEVICE}...")
        self.model = WhisperModel(
            STT_MODEL_SIZE,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )
        # Several utterances can be handled from different threads — serialize model access
        self._lock = threading.Lock()
        print("  [STT] Model loaded successfully.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: numpy float32 array of audio samples at 16kHz mono.

        Returns:
            Transcribed text string.
        """
        if audio is None or audio.size == 0:
            return ""

        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        start_time = time.perf_counter()

        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                beam_size=STT_BEAM_SIZE,
                language=STT_LANGUAGE,
                vad_filter=False,             # Outer Silero VAD already accurately segmented speech
                without_timestamps=True,       # Faster without timestamps
                condition_on_previous_text=False,  # Prevents hallucination loops
            )

            # `segments` is a lazy generator — consume it while the lock is held
            text_parts = [segment.text.strip() for segment in segments]

        full_text = " ".join(text_parts).strip()
        elapsed = (time.perf_counter() - start_time) * 1000

        if full_text:
            print(f'  [STT] "{full_text}" ({elapsed:.0f}ms)')
        else:
            print(f"  [STT] (no speech detected, {elapsed:.0f}ms)")

        return full_text
