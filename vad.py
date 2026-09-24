"""
Silero VAD — Voice Activity Detection
Detects when the user starts and stops speaking in real-time.
"""

import torch
import numpy as np
from collections import deque
from config import (
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    AUDIO_SAMPLE_RATE,
)


class VoiceActivityDetector:
    """Real-time voice activity detector using Silero VAD v5."""

    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        # State tracking
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._audio_buffer = deque()

        # Pre-compute frame thresholds
        # Silero VAD works on 512-sample chunks at 16kHz (32ms per chunk)
        self._chunk_ms = 32  # 512 samples / 16000 Hz * 1000
        self._min_speech_chunks = max(1, VAD_MIN_SPEECH_MS // self._chunk_ms)
        self._min_silence_chunks = max(1, VAD_MIN_SILENCE_MS // self._chunk_ms)
        self._pad_chunks = max(0, VAD_SPEECH_PAD_MS // self._chunk_ms)

        print(f"  [VAD] Silero VAD loaded (threshold={VAD_THRESHOLD})")

    def reset(self):
        """Reset VAD state for a new utterance."""
        self.model.reset_states()
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._audio_buffer.clear()

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a 512-sample audio chunk (32ms at 16kHz).

        Args:
            audio_chunk: numpy array of float32 audio samples, shape (512,)

        Returns:
            dict with keys:
                - 'is_speech': bool — whether this chunk contains speech
                - 'confidence': float — VAD confidence [0, 1]
                - 'event': str or None — 'speech_start', 'speech_end', or None
        """
        # Ensure correct format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize if needed (ensure [-1, 1] range)
        max_val = np.abs(audio_chunk).max()
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        # Run Silero VAD
        tensor = torch.from_numpy(audio_chunk)
        with torch.no_grad():
            confidence = self.model(tensor, AUDIO_SAMPLE_RATE).item()

        is_speech = confidence >= VAD_THRESHOLD
        event = None

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1
            self._audio_buffer.append(audio_chunk.copy())

            # Transition: silence → speech
            if not self._is_speaking and self._speech_frames >= self._min_speech_chunks:
                self._is_speaking = True
                event = "speech_start"
        else:
            self._speech_frames = 0
            if self._is_speaking:
                self._silence_frames += 1
                self._audio_buffer.append(audio_chunk.copy())  # Keep padding

                # Transition: speech → silence
                if self._silence_frames >= self._min_silence_chunks:
                    self._is_speaking = False
                    event = "speech_end"
            else:
                # Not speaking — keep a small lookback buffer for padding
                self._audio_buffer.append(audio_chunk.copy())
                if len(self._audio_buffer) > self._pad_chunks + 2:
                    self._audio_buffer.popleft()

        return {
            "is_speech": is_speech,
            "confidence": confidence,
            "event": event,
        }

    def get_speech_audio(self) -> np.ndarray:
        """
        Retrieve the buffered speech audio after a 'speech_end' event.
        Automatically resets the buffer.

        Returns:
            numpy array of float32 audio containing the full utterance.
        """
        if not self._audio_buffer:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(list(self._audio_buffer))
        self._audio_buffer.clear()
        return audio

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
