# voice/voice_module.py

import threading
import time

from core.module import Module

from .microphone_engine import MicrophoneEngine
from .wakeword_engine import WakeWordEngine
from .vad_engine import VADEngine
from .recognition_engine import RecognitionEngine


class VoiceModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="Voice",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.microphone = MicrophoneEngine()
        self.wakeword = WakeWordEngine()
        self.vad = VADEngine()
        self.recognition = RecognitionEngine()

        self._running = False
        self._thread = None
        self.conversation_timeout = 8.0
        self._conversation_active = False
        self._manual_conversation = False
        self._last_interaction = 0.0
        self._tts_active = False

    def initialize(self):
        print("[Voice] Initializing...", flush=True)

        self.event_bus.subscribe(
            "conversation_mode_set",
            self.on_conversation_mode_set,
        )
        self.event_bus.subscribe("assistant_sentence", self._on_assistant_sentence)
        self.event_bus.subscribe("speech_started", self._on_speech_started)
        self.event_bus.subscribe("speech_finished", self._on_speech_finished)

        self._running = True
        self.microphone.start()

        self._thread = threading.Thread(
            target=self._listen_loop,
            name="VoiceListenLoop",
            daemon=True,
        )
        self._thread.start()

        print("[Voice] Ready", flush=True)

    def shutdown(self):
        print("[Voice] Shutting down...", flush=True)

        self.event_bus.unsubscribe(
            "conversation_mode_set",
            self.on_conversation_mode_set,
        )
        self.event_bus.unsubscribe("assistant_sentence", self._on_assistant_sentence)
        self.event_bus.unsubscribe("speech_started", self._on_speech_started)
        self.event_bus.unsubscribe("speech_finished", self._on_speech_finished)

        self._running = False

        try:
            self.microphone.stop()
        except Exception as exc:
            print(
                f"[Voice] Microphone shutdown error: {type(exc).__name__}: {exc}",
                flush=True,
            )

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None
        print("[Voice] Stopped", flush=True)

    def on_conversation_mode_set(self, enabled):
        self._manual_conversation = bool(enabled)

        if self._manual_conversation:
            self._conversation_active = True
            self._last_interaction = time.monotonic()
            print("[Voice] Conversation mode: ON (manual)", flush=True)
        else:
            self._conversation_active = False
            self._last_interaction = 0.0
            # Discard anything captured while processing the OFF command.
            self.microphone.clear_buffer()
            print("[Voice] Conversation mode: OFF", flush=True)

    def _start_conversation(self):
        self._conversation_active = True
        self._last_interaction = time.monotonic()
        print("[Voice] Conversation mode: ACTIVE", flush=True)

    def _conversation_expired(self):
        return (
            self._conversation_active
            and not self._manual_conversation
            and (time.monotonic() - self._last_interaction > self.conversation_timeout)
        )

    def _on_assistant_sentence(self, *args, **kwargs):
        # Suppress recognition as soon as ASTA queues speech. This closes the
        # race where the voice loop has already entered wake-word detection
        # before the audio playback worker emits speech_started.
        self._tts_active = True

    def _on_speech_started(self, *args, **kwargs):
        self._tts_active = True

    def _on_speech_finished(self, *args, **kwargs):
        self._tts_active = False
        # Remove TTS echo/residual audio before wake-word detection resumes.
        self.microphone.clear_buffer()

    def _can_listen(self):
        return self._running and not self._tts_active

    def _listen_loop(self):
        while self._running:
            try:
                if not self._can_listen():
                    time.sleep(0.05)
                    continue

                if not self._conversation_active:
                    initial_audio = self.wakeword.wait_for_wakeword(
                        self.microphone,
                        should_continue=self._can_listen,
                    )

                    if not self._running:
                        break

                    if initial_audio is None or not self._can_listen():
                        continue

                    self._start_conversation()
                else:
                    initial_audio = None

                    if self._conversation_expired():
                        self._conversation_active = False
                        print(
                            "[Voice] Conversation mode: INACTIVE",
                            flush=True,
                        )
                        self.microphone.clear_buffer()
                        continue

                if not self._can_listen():
                    continue

                audio = self.vad.collect_utterance(
                    self.microphone,
                    initial_audio,
                    speech_timeout=3,
                )

                if not self._running:
                    break

                if not self._can_listen():
                    continue

                if audio is None:
                    if self._conversation_expired():
                        self._conversation_active = False
                        print(
                            "[Voice] Conversation mode: INACTIVE",
                            flush=True,
                        )
                        self.microphone.clear_buffer()
                    continue

                if not self._can_listen():
                    continue

                text = self.recognition.transcribe(audio)
                if not text:
                    continue

                print(f"[Voice] User: {text}", flush=True)
                self._last_interaction = time.monotonic()

                self.event_bus.emit("user_message", text=text)

                self._last_interaction = time.monotonic()

            except Exception as exc:
                print(
                    f"[Voice] Error: {type(exc).__name__}: {exc}",
                    flush=True,
                )

                if self._running:
                    threading.Event().wait(0.1)
