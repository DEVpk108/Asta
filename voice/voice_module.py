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
        self._last_interaction = 0.0

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def initialize(self):
        print("[Voice] Initializing...")

        self._running = True

        self.microphone.start()

        self._thread = threading.Thread(
            target=self._listen_loop,
            name="VoiceListenLoop",
            daemon=True,
        )

        self._thread.start()

        print("[Voice] Ready")

    def shutdown(self):
        print("[Voice] Shutting down...")

        self._running = False

        # Stop microphone first so get_chunk() can unblock.
        try:
            self.microphone.stop()
        except Exception as exc:
            print(
                f"[Voice] Microphone shutdown error: "
                f"{type(exc).__name__}: {exc}"
            )

        # Wait briefly for the worker thread to finish.
        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            self._thread.join(timeout=2.0)

        self._thread = None

        print("[Voice] Stopped")

    # ---------------------------------------------------------
    # Main voice loop
    # ---------------------------------------------------------
    
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
            and (
                time.monotonic()
                - self._last_interaction
                > self.conversation_timeout
            )
        )   
    
    

    def _listen_loop(self):

        while self._running:

            try:
                # -------------------------------------------------
                # 1. Standby mode: wait for a wake word.
                # -------------------------------------------------

                if not self._conversation_active:

                    initial_audio = (
                        self.wakeword.wait_for_wakeword(
                            self.microphone
                        )
                    )

                    if not self._running:
                        break

                    self._start_conversation()

                else:
                    # -------------------------------------------------
                    # 2. Conversation mode:
                    #    no wake word required.
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

                # -------------------------------------------------
                # No speech after wake word / during conversation.
                # -------------------------------------------------

                if audio is None:

                    if self._conversation_expired():
                        self._conversation_active = False

                        print(
                            "[Voice] Conversation mode: INACTIVE",
                            flush=True,
                        )

                    continue

                # -------------------------------------------------
                # 4. Speech → text via Moonshine.
                # -------------------------------------------------

                text = self.recognition.transcribe(
                    audio
                )

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