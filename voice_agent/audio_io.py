"""
Microphone listener — shared by agent.py and bruce.py.

The sounddevice callback only copies audio into a queue (it must return in microseconds).
All Silero VAD work happens in a worker thread, so slow inference can no longer cause
input overflows. The listener can also be MUTED (echo gate) while the agent is speaking.
"""

import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from core.config import AUDIO_BLOCKSIZE, AUDIO_CHANNELS, AUDIO_SAMPLE_RATE

_CHUNK = 512  # Silero VAD frame size at 16 kHz


class MicListener:
    def __init__(
        self,
        vad,
        on_speech: Callable[[np.ndarray], None],
        is_muted: Callable[[], bool] | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self.vad = vad
        self._on_speech = on_speech
        self._is_muted = is_muted or (lambda: False)
        self._on_speech_start = on_speech_start
        self._on_error = on_error

        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._stream = None

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Open the microphone. Raises if no input device can be opened."""
        self._stop.clear()
        self._stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            blocksize=AUDIO_BLOCKSIZE,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    # ─── internals ────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        try:
            self._q.put_nowait(indata[:, 0].copy().astype(np.float32))
        except queue.Full:
            pass  # drop audio rather than block the audio thread

    def _report(self, msg: str):
        if self._on_error:
            try:
                self._on_error(msg)
            except Exception:
                pass

    def _run(self):
        was_muted = False
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            muted = bool(self._is_muted())
            if muted:
                if not was_muted:
                    self.vad.reset()          # drop any half-heard echo
                was_muted = True
                continue
            if was_muted:
                self.vad.reset()
                was_muted = False

            for i in range(0, len(chunk), _CHUNK):
                sub = chunk[i:i + _CHUNK]
                if len(sub) < _CHUNK:
                    sub = np.pad(sub, (0, _CHUNK - len(sub)))
                try:
                    result = self.vad.process_chunk(sub)
                except Exception as e:
                    self._report(f"VAD error: {e}")
                    continue

                if result["event"] == "speech_start" and self._on_speech_start:
                    try:
                        self._on_speech_start()
                    except Exception:
                        pass

                if result["event"] == "speech_end":
                    audio = self.vad.get_speech_audio()
                    self.vad.reset()
                    if len(audio) > 1600:     # at least 100 ms
                        threading.Thread(target=self._on_speech, args=(audio,), daemon=True).start()
