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

        self.conversation_timeout = 30.0
        self._conversation_active = False
        self._manual_conversation = False
        self._last_interaction = 0.0
        self._tts_active = False
        self._tts_guard_until = 0.0
        self._microphone_paused_for_tts = False

        # A pending high-risk tool approval needs a more sensitive listener
        # because responses like "yes" and "no" are intentionally very short.
        self._awaiting_confirmation = False

        # Ignore an identical transcript captured again immediately after a
        # command. This protects against residual audio / VAD edge cases while
        # preserving legitimate repeated commands after a short pause.
        self._last_transcript = ""
        self._last_transcript_at = 0.0
        self._duplicate_window = 1.5

        self._wakeword_greeting_used = False
        self.wakeword_first_greeting = "Hello! How can I help?"
        self.wakeword_return_greeting = "Yes?"

    def initialize(self):
        print("[Voice] Initializing...", flush=True)

        self.event_bus.subscribe("conversation_mode_set", self.on_conversation_mode_set)
        self.event_bus.subscribe("assistant_sentence", self._on_assistant_sentence)
        self.event_bus.subscribe("speech_started", self._on_speech_started)
        self.event_bus.subscribe("speech_finished", self._on_speech_finished)
        self.event_bus.subscribe(
            "tool_confirmation_required",
            self._on_tool_confirmation_required,
        )
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self._on_tool_confirmation_response,
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

        self.event_bus.unsubscribe("conversation_mode_set", self.on_conversation_mode_set)
        self.event_bus.unsubscribe("assistant_sentence", self._on_assistant_sentence)
        self.event_bus.unsubscribe("speech_started", self._on_speech_started)
        self.event_bus.unsubscribe("speech_finished", self._on_speech_finished)
        self.event_bus.unsubscribe(
            "tool_confirmation_required",
            self._on_tool_confirmation_required,
        )
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self._on_tool_confirmation_response,
        )

        self._running = False

        try:
            self.microphone.stop()
            self.microphone.close()
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
            self._last_transcript = ""
            self._last_transcript_at = 0.0
            print("[Voice] Conversation mode: ON (manual)", flush=True)
        else:
            self._conversation_active = False
            self._last_interaction = 0.0
            self._last_transcript = ""
            self._last_transcript_at = 0.0
            self._awaiting_confirmation = False
            self.microphone.clear_buffer()
            print("[Voice] Conversation mode: OFF", flush=True)

    def _start_conversation(self):
        self._conversation_active = True
        self._last_interaction = time.monotonic()
        print("[Voice] Conversation mode: ACTIVE", flush=True)

        if self._wakeword_greeting_used:
            greeting = self.wakeword_return_greeting
        else:
            greeting = self.wakeword_first_greeting
            self._wakeword_greeting_used = True

        if greeting:
            self.event_bus.emit("assistant_sentence", text=greeting)

    def _conversation_expired(self):
        return (
            self._conversation_active
            and not self._manual_conversation
            and (time.monotonic() - self._last_interaction > self.conversation_timeout)
        )

    def _on_assistant_sentence(self, *args, **kwargs):
        self._tts_active = True
        self._tts_guard_until = time.monotonic() + 0.20
        self.microphone.clear_buffer()

    def _on_speech_started(self, *args, **kwargs):
        self._tts_active = True
        self._tts_guard_until = time.monotonic() + 0.20
        self.microphone.clear_buffer()

        if not self._microphone_paused_for_tts:
            try:
                self.microphone.stop()
                self._microphone_paused_for_tts = True
                print("[Voice] Microphone paused during TTS", flush=True)
            except Exception as exc:
                print(
                    f"[Voice] Microphone pause error: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _on_speech_finished(self, *args, **kwargs):
        self._tts_active = False
        # Give the microphone a little settling time after playback so the
        # first VAD window cannot start on residual device/room audio.
        self._tts_guard_until = time.monotonic() + 0.60
        self.microphone.clear_buffer()

        if self._microphone_paused_for_tts and self._running:
            try:
                self.microphone.start()
                self._microphone_paused_for_tts = False
                print("[Voice] Microphone resumed after TTS", flush=True)
            except Exception as exc:
                print(
                    f"[Voice] Microphone resume error: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        if self._conversation_active and not self._manual_conversation:
            self._last_interaction = time.monotonic()

    def _on_tool_confirmation_required(self, *args, **kwargs):
        self._awaiting_confirmation = True
        self._last_transcript = ""
        self._last_transcript_at = 0.0
        print("[Voice] Confirmation listening: ENABLED", flush=True)

    def _on_tool_confirmation_response(self, *args, **kwargs):
        self._awaiting_confirmation = False
        print("[Voice] Confirmation listening: DISABLED", flush=True)

    def _can_listen(self):
        return (
            self._running
            and not self._tts_active
            and not self._microphone_paused_for_tts
            and time.monotonic() >= self._tts_guard_until
        )

    def _is_duplicate_transcript(self, text):
        normalized = " ".join(text.lower().split())
        now = time.monotonic()
        if (
            normalized
            and normalized == self._last_transcript
            and now - self._last_transcript_at < self._duplicate_window
        ):
            print(f"[STT] Rejected duplicate transcript: {text!r}", flush=True)
            return True

        self._last_transcript = normalized
        self._last_transcript_at = now
        return False

    def _collect_command_audio(self):
        if not self._awaiting_confirmation:
            return self.vad.collect_utterance(
                self.microphone,
                speech_timeout=3,
            )

        # Confirmation replies are deliberately short and quieter than normal
        # commands. Temporarily relax only the VAD acceptance thresholds while
        # an approval decision is pending, then restore the normal profile.
        original = {
            "min_rms": self.vad.min_rms,
            "min_peak": self.vad.min_peak,
            "start_chunk_rms": self.vad.start_chunk_rms,
            "min_speech_duration": self.vad.min_speech_duration,
        }
        self.vad.min_rms = 0.010
        self.vad.min_peak = 0.035
        self.vad.start_chunk_rms = 0.003
        self.vad.min_speech_duration = 0.20

        try:
            print("[VAD] Listening for short confirmation...", flush=True)
            return self.vad.collect_utterance(
                self.microphone,
                speech_timeout=2.0,
            )
        finally:
            self.vad.min_rms = original["min_rms"]
            self.vad.min_peak = original["min_peak"]
            self.vad.start_chunk_rms = original["start_chunk_rms"]
            self.vad.min_speech_duration = original["min_speech_duration"]

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
                    if self._conversation_expired():
                        self._conversation_active = False
                        print("[Voice] Conversation mode: INACTIVE", flush=True)
                        self.microphone.clear_buffer()
                        continue

                if not self._can_listen():
                    continue

                # Wake-word detection only activates the conversation. Command
                # recognition starts from fresh microphone audio after this point.
                self.microphone.clear_buffer()

                audio = self._collect_command_audio()

                if not self._running:
                    break

                if not self._can_listen():
                    continue

                if audio is None:
                    if self._conversation_expired():
                        self._conversation_active = False
                        print("[Voice] Conversation mode: INACTIVE", flush=True)
                        self.microphone.clear_buffer()
                    continue

                if not self._can_listen():
                    continue

                text = self.recognition.transcribe(audio)
                if not text:
                    continue

                if self._is_duplicate_transcript(text):
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
                time.sleep(0.1)
