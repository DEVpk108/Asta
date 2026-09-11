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

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def initialize(self):
        print("[Voice] Initializing...", flush=True)

        self.event_bus.subscribe(
            "conversation_mode_set",
            self.on_conversation_mode_set,
        )

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

        self._running = False

        try:
            self.microphone.stop()
        except Exception as exc:
            print(
                f"[Voice] Microphone shutdown error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            self._thread.join(timeout=2.0)

        self._thread = None

        print("[Voice] Stopped", flush=True)

    # ---------------------------------------------------------
    # Conversation mode controls
    # ---------------------------------------------------------

    def on_conversation_mode_set(self, enabled):
        self._manual_conversation = bool(enabled)

        if self._manual_conversation:
            self._conversation_active = True
            self._last_interaction = time.monotonic()
            print("[Voice] Conversation mode: ON (manual)", flush=True)
        else:
            self._conversation_active = False
            self._last_interaction = 0.0
            print("[Voice] Conversation mode: OFF", flush=True)

    def _start_conversation(self):
        self._conversation_active = True
        self._last_interaction = time.monotonic()

        print(
            "[Voice] Conversation mode: ACTIVE",
            flush=True,
        )

    def _conversation_expired(self):
        return (
            self._conversation_active
            and not self._manual_conversation
            and (
                time.monotonic()
                - self._last_interaction
                > self.conversation_timeout
            )
        )

    # ---------------------------------------------------------
    # Main voice loop
    # ---------------------------------------------------------

    def _listen_loop(self):
        while self._running:
            try:
                # -------------------------------------------------
                # 1. Standby mode: wait for a wake word.
                # -------------------------------------------------
                if not self._conversation_active:
                    initial_audio = self.wakeword.wait_for_wakeword(
                        self.microphone
                    )

                    if not self._running:
                        break

                    self._start_conversation()
                else:
                    # -------------------------------------------------
                    # 2. Conversation mode: no wake word required.
                    # -------------------------------------------------
                    initial_audio = None

                    if self._conversation_expired():
                        self._conversation_active = False

                        print(
                            "[Voice] Conversation mode: INACTIVE",
                            flush=True,
                        )

                        continue

                # -------------------------------------------------
                # 3. Capture the user's command.
                # -------------------------------------------------
                audio = self.vad.collect_utterance(
                    self.microphone,
                    initial_audio,
                    speech_timeout=3,
                )

                if not self._running:
                    break

                if audio is None:
                    if self._conversation_expired():
                        self._conversation_active = False

                        print(
                            "[Voice] Conversation mode: INACTIVE",
                            flush=True,
                        )

                    continue

                # -------------------------------------------------
                # 4. Speech -> text.
                # -------------------------------------------------
                text = self.recognition.transcribe(audio)

                if not text:
                    continue

                print(
                    f"[Voice] User: {text}",
                    flush=True,
                )

                # -------------------------------------------------
                # 5. Refresh conversation timeout.
                # -------------------------------------------------
                self._last_interaction = time.monotonic()

                # -------------------------------------------------
                # 6. Send recognized text into ASTA.
                # -------------------------------------------------
                self.event_bus.emit(
                    "user_message",
                    text=text,
                )

                # -------------------------------------------------
                # 7. Refresh timeout after processing.
                # -------------------------------------------------
                self._last_interaction = time.monotonic()

            except Exception as exc:
                print(
                    f"[Voice] Error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                if self._running:
                    threading.Event().wait(0.1)
