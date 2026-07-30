from .recognition_engine import RecognitionEngine
from .microphone_engine import MicrophoneEngine
from .vad_engine import VADEngine
from .wakeword_engine import WakeWordEngine



mic = MicrophoneEngine()
vad = VADEngine()
rec = RecognitionEngine()
wake = WakeWordEngine()

mic.start()
try:
    while True:
        try:
            initial_audio = wake.wait_for_wakeword(mic)

            audio = vad.collect_utterance(
                mic,
                initial_audio,
                speech_timeout=3,
            )

            if audio is None:
                continue

            text = rec.transcribe(audio)
            print(text)

        except Exception as e:
            print(f"[Voice] {type(e).__name__}: {e}")

except KeyboardInterrupt:
    print("\nStopping voice test...")