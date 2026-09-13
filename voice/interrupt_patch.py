"""Runtime patch for reliable voice barge-in detection.

The main VoiceModule already owns the conversation lifecycle. This helper only
replaces its barge-in recognizer with a non-consuming microphone-ring-buffer
listener so interruption detection cannot fight the normal VAD/command reader.
"""

import time
from collections import deque

import numpy as np

from .voice_module import VoiceModule


INTERRUPT_PHRASES = (
    "okay okay stop",
    "okay stop",
    "ok stop",
    "okay please stop",
    "ok please stop",
    "please stop",
    "stop asta",
    "stop asta now",
    "asta stop",
    "wait asta",
    "hold on asta",
    "stop speaking",
    "stop talking",
    "stop presenting",
    "stop presentation",
    "stop the presentation",
    "hold on",
)


def _extract_interrupt_tail(text):
    normalized = " ".join(str(text).strip().lower().split())
    normalized = normalized.rstrip(" .!?;:,")
    if not normalized:
        return False, ""

    for phrase in INTERRUPT_PHRASES:
        if normalized == phrase:
            return True, ""
        if normalized.startswith(phrase + " "):
            return True, normalized[len(phrase):].strip(" ,.-")

    # Whisper often inserts one or two words between the acknowledgement and
    # "stop", e.g. "okay please stop" or "okay okay please stop".
    tokens = normalized.split()
    if "stop" in tokens:
        stop_index = tokens.index("stop")
        before = tokens[:stop_index]
        after = tokens[stop_index + 1 :]

        if stop_index == 0:
            if not after or after[0] in {
                "speaking", "talking", "presenting", "presentation"
            }:
                return True, ""

        recent = before[-4:]
        if any(token in {"okay", "ok", "please", "asta", "wait", "hold"} for token in recent):
            return True, " ".join(after).strip(" ,.-")

    # A bare "stop" is intentionally accepted while A.S.T.A. is speaking.
    if normalized == "stop":
        return True, ""

    return False, ""


def _barge_listen_loop(self):
    sample_rate = self.microphone.sample_rate
    max_samples = int(sample_rate * 2.0)
    min_samples = int(sample_rate * 0.55)
    check_interval = 0.25
    min_rms = 0.006

    # Preserve the existing audio already captured by the microphone, but keep
    # up to two seconds so phrases such as "okay, okay, stop" are fully visible.
    try:
        current = self.microphone.get_buffer()
        self.microphone.ring_buffer = deque(
            np.asarray(current[-max_samples:], dtype=np.float32),
            maxlen=max_samples,
        )
    except Exception:
        pass

    next_check = time.monotonic() + 0.15
    last_checked_text = ""

    while self._running and not self._barge_stop.is_set():
        if self._barge_stop.wait(0.05):
            break

        now = time.monotonic()
        if now < next_check:
            continue
        next_check = now + check_interval

        try:
            audio = self.microphone.get_buffer()
        except Exception:
            continue

        if audio.size < min_samples:
            continue

        audio = np.asarray(audio[-max_samples:], dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        if rms < min_rms:
            continue

        try:
            text = self.recognition.transcribe(audio)
        except Exception as exc:
            print(
                f"[Voice] Barge-in STT error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        if not text or text == last_checked_text:
            continue

        last_checked_text = text
        print(f"[Interrupt] Barge transcript: {text!r}", flush=True)

        interrupted, tail = _extract_interrupt_tail(text)
        if not interrupted:
            continue

        print(f"[Interrupt] Barge-in detected: {text!r}", flush=True)
        self.event_bus.emit("speech_interrupt", text=text)

        if tail:
            print(f"[Interrupt] Continuing with: {tail}", flush=True)
            self._last_interaction = time.monotonic()
            self.event_bus.emit("user_message", text=tail)

        break


def apply_voice_interrupt_patch():
    """Patch the runtime VoiceModule barge-in listener before app startup."""
    VoiceModule._extract_interrupt_tail = staticmethod(_extract_interrupt_tail)
    VoiceModule._barge_listen_loop = _barge_listen_loop
