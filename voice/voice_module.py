import re
import threading
import time
from collections import deque

import numpy as np

from core.module import Module

from .microphone_engine import MicrophoneEngine
from .wakeword_engine import WakeWordEngine
from .vad_engine import VADEngine
from .recognition_engine import RecognitionEngine


class VoiceModule(Module):

    INTERRUPT_PHRASES = (
        "okay okay stop",
        "okay stop",
        "ok stop",
        "stop asta",
        "stop asta now",
        "asta stop",
        "wait asta",
        "hold on asta",
        "stop speaking",
        "stop talking",
        "stop presentation",
        "stop the presentation",
        "hold on",
    )

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
        self._barge_thread = None
        self._barge_stop = threading.Event()
        self._barge_check_lock = threading.Lock()
        self._listener_ready_reported = False

        self.conversation_timeout = 30.0
        self._conversation_active = False
        self._manual_conversation = False
        self._last_interaction = 0.0
        self._tts_active = False
        self._tts_guard_until = 0.0
        self._microphone_paused_for_tts = False

        self._barge_window_seconds = 1.4
        self._barge_check_interval = 0.65
        self._barge_min_rms = 0.010

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
        self.event_bus.subscribe("speech_interrupt", self._on_speech_interrupt)
        self.event_bus.subscribe(
            "tool_confirmation_required",
            self._on_tool_confirmation_required,
        )
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self._on_tool_confirmation_response,
        )

        self._running = True
        self._listener_ready_reported = False
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
        self.event_bus.unsubscribe("speech_interrupt", self._on_speech_interrupt)
        self.event_bus.unsubscribe(
            "tool_confirmation_required",
            self._on_tool_confirmation_required,
        )
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self._on_tool_confirmation_response,
        )

        self._running = False
        self._stop_barge_listener()

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
            microphone = getattr(self, "microphone", None)
            clear_buffer = getattr(microphone, "clear_buffer", None)
            if callable(clear_buffer):
                clear_buffer()
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
        self._start_barge_listener()
        print("[Voice] Barge-in listening: ENABLED", flush=True)

    def _on_speech_finished(self, *args, **kwargs):
        self._tts_active = False
        self._tts_guard_until = time.monotonic() + 0.60
        self._stop_barge_listener()
        self.microphone.clear_buffer()
        print("[Voice] Barge-in listening: DISABLED", flush=True)

        if self._conversation_active and not self._manual_conversation:
            self._last_interaction = time.monotonic()

    def _on_speech_interrupt(self, *args, **kwargs):
        print("[Voice] Speech interrupt received.", flush=True)
        self._tts_active = False
        self._tts_guard_until = time.monotonic() + 0.05
        self.microphone.clear_buffer()
        self._stop_barge_listener()

    def _start_barge_listener(self):
        with self._barge_check_lock:
            if (
                self._barge_thread is not None
                and self._barge_thread.is_alive()
            ):
                return

            self._barge_stop.clear()
            self._barge_thread = threading.Thread(
                target=self._barge_listen_loop,
                name="VoiceBargeInWorker",
                daemon=True,
            )
            self._barge_thread.start()

    def _stop_barge_listener(self):
        self._barge_stop.set()
        thread = self._barge_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        if thread is threading.current_thread():
            return
        self._barge_thread = None

    @classmethod
    def _extract_interrupt_tail(cls, text):
        normalized = " ".join(str(text).strip().lower().split())
        normalized = normalized.rstrip(" .!?;:,")
        if not normalized:
            return False, ""

        if normalized in {"okay, okay, stop", "okay okay stop", "ok, ok, stop", "ok ok stop"}:
            return True, ""

        if re.match(r"^(?:(?:okay|ok|please)\s*,?\s*)+stop$", normalized):
            return True, ""

        for phrase in cls.INTERRUPT_PHRASES:
            marker = phrase
            if normalized == marker:
                return True, ""
            if normalized.startswith(marker + " "):
                tail = normalized[len(marker):].strip(" ,.-")
                return True, tail
            if normalized.endswith(" " + marker):
                return True, ""

        if normalized == "stop":
            return True, ""

        # Whisper may insert acknowledgements before the stop word, e.g.
        # "okay, okay, stop" or "okay please stop". Treat recent conversational
        # filler before a bare stop as an interruption while preserving the tail.
        tokens = normalized.split()
        if "stop" in tokens:
            stop_index = tokens.index("stop")
            before = tokens[:stop_index]
            after = tokens[stop_index + 1:]
            if not after or after[0] in {"speaking", "talking", "presenting", "presentation"}:
                recent = before[-4:]
                if any(token in {"okay", "ok", "please", "asta", "wait", "hold"} for token in recent):
                    return True, " ".join(after).strip(" ,.-")

        if normalized.startswith("stop "):
            remainder = normalized[5:].strip()
            if remainder in {"speaking", "talking", "presenting", "presentation"}:
                return True, ""

        return False, ""

    def _barge_listen_loop(self):
        max_samples = int(self.microphone.sample_rate * self._barge_window_seconds)
        rolling = deque(maxlen=max_samples)
        next_check = time.monotonic() + self._barge_check_interval

        while self._running and not self._barge_stop.is_set():
            try:
                chunk = self.microphone.get_chunk().flatten()
            except Exception:
                if self._barge_stop.wait(0.02):
                    break
                continue

            if chunk is None or len(chunk) == 0:
                continue

            rolling.extend(np.asarray(chunk, dtype=np.float32))
            now = time.monotonic()
            if now < next_check:
                continue
            next_check = now + self._barge_check_interval

            if len(rolling) < int(self.microphone.sample_rate * 0.55):
                continue

            audio = np.asarray(rolling, dtype=np.float32)
            rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            if rms < self._barge_min_rms:
                continue

            try:
                text = self.recognition.transcribe(audio, strict=True)
            except Exception as exc:
                print(
                    f"[Voice] Barge-in STT error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            if not text:
                continue

            interrupted, tail = self._extract_interrupt_tail(text)
            if not interrupted:
                continue

            print(f"[Interrupt] Barge-in detected: {text!r}", flush=True)
            self.event_bus.emit("speech_interrupt", text=text)

            if tail:
                print(f"[Interrupt] Continuing with: {tail}", flush=True)
                self._last_interaction = time.monotonic()
                self.event_bus.emit("user_message", text=tail)

            break

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

    @staticmethod
    def _normalize_confirmation_transcript(text):
        """Canonicalize common short approval/rejection phrases.

        Whisper may hear a natural reply such as "okay so", "yeah go ahead",
        or "no thanks" when the user is answering a yes/no approval prompt.
        During confirmation mode only, reduce these short replies to the
        decision word so the deterministic approval handler can act on them.
        """
        normalized = " ".join(str(text).strip().lower().split())
        normalized = normalized.rstrip(" .!?;:,")

        aliases = {
            "yes": ("yes", "yeah", "yep", "yup", "sure", "okay", "ok"),
            "no": ("no", "nope", "nah"),
            "cancel": ("cancel",),
            "confirm": ("confirm", "confirmed"),
        }

        tokens = normalized.split()
        if not tokens:
            return text

        first = tokens[0]
        for canonical, variants in aliases.items():
            if first in variants and len(tokens) <= 4:
                print(
                    f"[STT] Confirmation normalization: {text!r} -> {canonical!r}",
                    flush=True,
                )
                return canonical

        phrase_aliases = {
            "go ahead": "go ahead",
            "do it": "do it",
            "proceed": "proceed",
            "i confirm": "i confirm",
            "yes proceed": "yes proceed",
            "do not": "do not",
        }
        if normalized in phrase_aliases:
            return phrase_aliases[normalized]

        return text

    def _collect_command_audio(self):
        if not self._awaiting_confirmation:
            return self.vad.collect_utterance(
                self.microphone,
                speech_timeout=3,
            )

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
            self.vad.start_chunk_rms = original["min_rms"] if False else original["min_peak"]
            self.vad.start_chunk_rms = original["start_chunk_rms"]
            self.vad.min_speech_duration = original["min_speech_duration"]

    def _listen_loop(self):
        # This event is intentionally emitted from the actual listener worker,
        # not from initialize(). At this point the microphone is running and the
        # wake-word engine is ready to receive audio. The HUD uses this as its
        # final "A.S.T.A. READY" gate.
        if not self._listener_ready_reported:
            self._listener_ready_reported = True
            self.event_bus.emit("voice_ready")
            print("[Voice] Wake-word listener ready.", flush=True)

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

                if self._awaiting_confirmation:
                    text = self._normalize_confirmation_transcript(text)

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
