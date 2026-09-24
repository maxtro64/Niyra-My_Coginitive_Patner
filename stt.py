"""
Faster-Whisper STT — Speech to Text
CTranslate2 implementation of OpenAI Whisper for fast, accurate transcription.
"""

import numpy as np
from faster_whisper import WhisperModel
from config import (
    STT_MODEL_SIZE,
    STT_DEVICE,
    STT_COMPUTE_TYPE,
    STT_BEAM_SIZE,
    STT_LANGUAGE,
)
import time


class SpeechToText:
    """Fast, GPU-accelerated speech transcription using Faster-Whisper."""

    def __init__(self):
        print(f"  [STT] Loading Faster-Whisper '{STT_MODEL_SIZE}' on {STT_DEVICE}...")
        self.model = WhisperModel(
            STT_MODEL_SIZE,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )
        print(f"  [STT] Model loaded successfully.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: numpy float32 array of audio samples at 16kHz mono.

        Returns:
            Transcribed text string.
        """
        if audio.size == 0:
            return ""

        start_time = time.perf_counter()

        segments, info = self.model.transcribe(
            audio,
            beam_size=STT_BEAM_SIZE,
            language=STT_LANGUAGE,
            vad_filter=True,              # Use Silero VAD within Whisper too
            vad_parameters=dict(
                min_silence_duration_ms=300,
            ),
            without_timestamps=True,       # Faster without timestamps
            condition_on_previous_text=False,  # Prevents hallucination loops
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts).strip()
        elapsed = (time.perf_counter() - start_time) * 1000

        if full_text:
            print(f"  [STT] \"{full_text}\" ({elapsed:.0f}ms)")
        else:
            print(f"  [STT] (no speech detected, {elapsed:.0f}ms)")

        return full_text
