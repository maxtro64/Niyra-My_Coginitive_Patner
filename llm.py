"""
Ollama LLM — Local Language Model via Phi3 Mini
Streaming responses for low-latency sentence-by-sentence TTS.
"""

import ollama
from config import (
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
import time
from typing import Generator


class LanguageModel:
    """Local LLM interface using Ollama with streaming support."""

    def __init__(self):
        print(f"  [LLM] Connecting to Ollama ({LLM_MODEL})...")

        # Verify model is available
        try:
            models = ollama.list()
        except Exception:
            import subprocess, sys
            print("  [LLM] Ollama daemon not running. Auto-starting Ollama in background...")
            creation_flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creation_flags)
            except Exception:
                try:
                    subprocess.Popen([r"C:\Users\91639\AppData\Local\Programs\Ollama\ollama.exe", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creation_flags)
                except Exception as ex:
                    print(f"  [LLM] Could not auto-start Ollama: {ex}")
            time.sleep(3)

        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            found = any(LLM_MODEL.split(":")[0] in name for name in model_names)
            if found:
                print(f"  [LLM] Model '{LLM_MODEL}' is ready.")
            else:
                print(f"  [LLM] Model '{LLM_MODEL}' not found. Pulling...")
                ollama.pull(LLM_MODEL)
                print(f"  [LLM] Model pulled successfully.")
        except Exception as e:
            print(f"  [LLM] Warning: Could not verify model availability: {e}")

        # Conversation history for context
        self.history = []
        self._max_history = 10  # Keep last 10 exchanges for context

    def generate_streaming(self, user_text: str) -> Generator[str, None, None]:
        """
        Generate a response from the LLM, yielding complete sentences
        as they become available for immediate TTS synthesis.

        Args:
            user_text: The user's transcribed speech.

        Yields:
            Complete sentences as strings.
        """
        start_time = time.perf_counter()

        # Build messages with history
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        messages.extend(self.history[-self._max_history:])
        messages.append({"role": "user", "content": user_text})

        # Stream response from Ollama
        sentence_buffer = ""
        full_response = ""
        sentence_terminators = {'.', '!', '?'}
        # Also break on commas/semicolons for longer sentences
        soft_terminators = {',', ';', ':', '—', '–'}
        
        first_token_time = None

        try:
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                options={
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                    "num_ctx": 2048,       # Context window
                    "num_gpu": 99,         # Use all GPU layers
                },
            )

            for chunk in stream:
                token = chunk.message.content
                if not token:
                    continue

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                sentence_buffer += token
                full_response += token

                # Check for sentence boundaries
                stripped = sentence_buffer.strip()
                if not stripped:
                    continue

                last_char = stripped[-1]

                # Hard sentence end
                if last_char in sentence_terminators and len(stripped) > 10:
                    yield stripped
                    sentence_buffer = ""

                # Soft break for very long clauses (natural speech pauses)
                elif last_char in soft_terminators and len(stripped) > 60:
                    yield stripped
                    sentence_buffer = ""

            # Flush remaining buffer
            remaining = sentence_buffer.strip()
            if remaining:
                yield remaining

        except Exception as e:
            yield f"Sorry, I had a hiccup. {str(e)}"
            full_response = f"[error: {e}]"

        # Update conversation history
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})

        # Trim history
        if len(self.history) > self._max_history * 2:
            self.history = self.history[-self._max_history * 2:]

        elapsed = (time.perf_counter() - start_time) * 1000
        ttft = ((first_token_time - start_time) * 1000) if first_token_time else 0
        print(f"  [LLM] Response complete ({elapsed:.0f}ms total, {ttft:.0f}ms TTFT)")

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        print("  [LLM] Conversation history cleared.")
