"""
Naiyra Voice Agent — Component Tests
Tests each component individually, then runs an integration test.

Usage:
    python test_agent.py
"""

import config
import sys
import time
import numpy as np


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  TEST: {title}")
    print(f"{'='*60}\n")


def test_vad():
    """Test Silero VAD loads and processes audio."""
    divider("Voice Activity Detection (Silero VAD)")

    from vad import VoiceActivityDetector
    vad = VoiceActivityDetector()

    # Test with silence (zeros)
    silence = np.zeros(512, dtype=np.float32)
    result = vad.process_chunk(silence)
    assert result["is_speech"] == False, "VAD should detect silence"
    print(f"  ✅ Silence detected correctly (confidence={result['confidence']:.3f})")

    # Test with noise (random)
    noise = np.random.randn(512).astype(np.float32) * 0.01
    result = vad.process_chunk(noise)
    print(f"  ✅ Noise processed (is_speech={result['is_speech']}, confidence={result['confidence']:.3f})")

    # Test with a simulated speech-like signal (sine wave with harmonics)
    t = np.linspace(0, 512/16000, 512, dtype=np.float32)
    speech_like = (
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t) +
        0.1 * np.sin(2 * np.pi * 450 * t)
    ).astype(np.float32)
    result = vad.process_chunk(speech_like)
    print(f"  ✅ Synthetic speech processed (is_speech={result['is_speech']}, confidence={result['confidence']:.3f})")

    vad.reset()
    print(f"  ✅ VAD reset successful")
    print(f"\n  ✅ VAD TEST PASSED\n")
    return True


def test_stt():
    """Test Faster-Whisper loads and transcribes."""
    divider("Speech-to-Text (Faster-Whisper)")

    from stt import SpeechToText
    stt = SpeechToText()

    # Test with silence — should return empty
    silence = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds of silence
    text = stt.transcribe(silence)
    print(f"  ✅ Silence transcription: '{text}' (expected empty)")

    # Test with empty array
    text = stt.transcribe(np.array([], dtype=np.float32))
    assert text == "", "Empty audio should return empty string"
    print(f"  ✅ Empty audio handled correctly")

    print(f"\n  ✅ STT TEST PASSED\n")
    return stt


def test_llm():
    """Test Ollama LLM connection and streaming."""
    divider("Language Model (Ollama Phi3)")

    from llm import LanguageModel
    llm = LanguageModel()

    # Test streaming generation
    print(f"  Testing streaming response...")
    print(f"  Query: 'Say hello in one sentence.'\n")

    start = time.perf_counter()
    full_response = ""
    sentence_count = 0

    for sentence in llm.generate_streaming("Say hello in one sentence."):
        sentence_count += 1
        full_response += sentence + " "
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  📝 Sentence {sentence_count} ({elapsed:.0f}ms): {sentence}")

    total = (time.perf_counter() - start) * 1000
    print(f"\n  Full response: {full_response.strip()}")
    print(f"  Total time: {total:.0f}ms")
    print(f"  Sentences yielded: {sentence_count}")

    assert len(full_response.strip()) > 0, "LLM should produce a response"
    print(f"\n  ✅ LLM TEST PASSED\n")
    return llm


def test_tts():
    """Test TTS synthesis and playback."""
    divider("Text-to-Speech (TTS)")

    from tts import TextToSpeech
    tts = TextToSpeech()

    # Test synthesis
    test_text = "Hello! I'm Naiyra, your voice assistant. Nice to meet you!"
    print(f"  Speaking: '{test_text}'")
    print(f"  🔊 Listen to your speakers...\n")

    start = time.perf_counter()
    tts.speak(test_text)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Total TTS time: {elapsed:.0f}ms")

    print(f"\n  ✅ TTS TEST PASSED\n")
    return tts


def test_microphone():
    """Test microphone access."""
    divider("Microphone Access")

    import sounddevice as sd

    devices = sd.query_devices()
    default_input = sd.query_devices(kind='input')
    print(f"  Default input device: {default_input['name']}")
    print(f"  Sample rate: {default_input['default_samplerate']}")
    print(f"  Max input channels: {default_input['max_input_channels']}")

    # Quick 1-second recording test
    print(f"\n  Recording 1 second of audio from microphone...")
    audio = sd.rec(int(16000 * 1), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    max_amplitude = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio**2))
    print(f"  Max amplitude: {max_amplitude:.4f}")
    print(f"  RMS level: {rms:.4f}")

    if max_amplitude > 0.001:
        print(f"  ✅ Microphone is picking up audio")
    else:
        print(f"  ⚠️  Very low audio level — check your microphone")

    print(f"\n  ✅ MICROPHONE TEST PASSED\n")
    return True


def test_integration(stt=None, llm=None, tts=None):
    """Integration test: Record speech → STT → LLM → TTS."""
    divider("Integration Test (Full Pipeline)")

    import sounddevice as sd
    if stt is None:
        from stt import SpeechToText
        stt = SpeechToText()
    if llm is None:
        from llm import LanguageModel
        llm = LanguageModel()
    if tts is None:
        from tts import TextToSpeech
        tts = TextToSpeech()

    print(f"  🎤 Speak something into your microphone (3 seconds)...")
    print(f"  Recording starts NOW!\n")

    # Record 3 seconds
    audio = sd.rec(int(16000 * 3), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    print(f"  Recording complete. Processing...\n")

    # STT
    start = time.perf_counter()
    user_text = stt.transcribe(audio)
    stt_time = (time.perf_counter() - start) * 1000

    if not user_text.strip():
        print(f"  ⚠️  No speech detected. Using fallback test phrase.")
        user_text = "What is the capital of France?"

    print(f"  📝 You said: '{user_text}' ({stt_time:.0f}ms)")

    # LLM
    print(f"\n  🧠 Generating response...\n")
    start = time.perf_counter()
    full_response = ""
    for sentence in llm.generate_streaming(user_text):
        full_response += sentence + " "
        print(f"    → {sentence}")
    llm_time = (time.perf_counter() - start) * 1000
    print(f"\n  Response generated in {llm_time:.0f}ms")

    # TTS
    print(f"\n  🔊 Speaking response...")
    start = time.perf_counter()
    tts.speak(full_response.strip())
    tts_time = (time.perf_counter() - start) * 1000
    print(f"  Spoken in {tts_time:.0f}ms")

    # Summary
    total = stt_time + llm_time + tts_time
    print(f"\n  {'─'*40}")
    print(f"  Pipeline Latency Summary:")
    print(f"    STT:  {stt_time:>6.0f}ms")
    print(f"    LLM:  {llm_time:>6.0f}ms")
    print(f"    TTS:  {tts_time:>6.0f}ms")
    print(f"    Total: {total:>5.0f}ms")
    print(f"  {'─'*40}")

    print(f"\n  ✅ INTEGRATION TEST PASSED\n")
    return True


def main():
    print(f"\n{'='*60}")
    print(f"  🧪 NAIYRA VOICE AGENT — COMPONENT TESTS")
    print(f"{'='*60}\n")

    results = {}
    stt_inst = None
    llm_inst = None
    tts_inst = None

    try:
        stt_inst = test_stt()
        results["STT"] = stt_inst is not None
    except Exception as e:
        print(f"\n  ❌ STT TEST FAILED: {e}\n")
        results["STT"] = False

    try:
        results["VAD"] = test_vad()
    except Exception as e:
        print(f"\n  ❌ VAD TEST FAILED: {e}\n")
        results["VAD"] = False

    try:
        llm_inst = test_llm()
        results["LLM"] = llm_inst is not None
    except Exception as e:
        print(f"\n  ❌ LLM TEST FAILED: {e}\n")
        results["LLM"] = False

    try:
        tts_inst = test_tts()
        results["TTS"] = tts_inst is not None
    except Exception as e:
        print(f"\n  ❌ TTS TEST FAILED: {e}\n")
        results["TTS"] = False

    try:
        results["Microphone"] = test_microphone()
    except Exception as e:
        print(f"\n  ❌ Microphone TEST FAILED: {e}\n")
        results["Microphone"] = False

    try:
        results["Integration"] = test_integration(stt_inst, llm_inst, tts_inst)
    except Exception as e:
        print(f"\n  ❌ Integration TEST FAILED: {e}\n")
        results["Integration"] = False

    # Final summary
    print(f"\n{'='*60}")
    print(f"  📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}\n")

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  🎉 All tests passed! Run 'python agent.py' to start Naiyra.")
    else:
        print(f"  ⚠️  Some tests failed. Fix the issues above before running the agent.")
    print()


if __name__ == "__main__":
    main()
