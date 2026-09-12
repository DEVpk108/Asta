import queue
import threading
import time

from core.module import Module
from .kokoro_engine import KokoroEngine


class SpeechModule(Module):

    PRESENTATION_SHORT_TEXT = (
        "Hello Sir. I’m A.S.T.A., a local-first AI engineering assistant. "
        "I understand voice commands, reason about technical questions, and can interact with the computer through authorized tools. "
        "My architecture is modular, with voice input, AI reasoning, tool execution, approval handling, speech output, and a HUD connected through the kernel. "
        "My current local stack uses LM Studio, Whisper, and Kokoro. "
        "My long-term direction is to become a personal AI operating system with stronger memory, workflow awareness, proactive assistance, and specialized agents."
    )

    def __init__(self, kernel):
        super().__init__(
            name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = KokoroEngine()

        self._queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._running = False
        self._synth_thread = None
        self._play_thread = None
        self.coalesce_window = 0.08

        self._state_lock = threading.Lock()
        self._speech_active = False
        self._queued_text = 0
        self._pending_audio = 0
        self._synthesis_inflight = 0

    def initialize(self):
        print("[Speech] Initializing...", flush=True)
        self._running = True
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)

        self._synth_thread = threading.Thread(
            target=self._synthesis_loop,
            name="SpeechSynthWorker",
            daemon=True,
        )
        self._play_thread = threading.Thread(
            target=self._playback_loop,
            name="SpeechPlaybackWorker",
            daemon=True,
        )
        self._synth_thread.start()
        self._play_thread.start()
        print("[Speech] Ready", flush=True)

    def shutdown(self):
        print("[Speech] Shutting down...", flush=True)
        self._running = False
        self.event_bus.unsubscribe("assistant_sentence", self.on_assistant_sentence)

        self._queue.put(None)
        self._audio_queue.put(None)

        if self._synth_thread is not None and self._synth_thread.is_alive():
            self._synth_thread.join(timeout=2.0)
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=2.0)

        self._synth_thread = None
        self._play_thread = None
        print("[Speech] Stopped", flush=True)

    def on_assistant_sentence(self, text):
        if not text:
            return

        queued_text = text
        if len(text) > 800 and text.startswith("Hello Sir. I’m A.S.T.A."):
            queued_text = self.PRESENTATION_SHORT_TEXT
            print("[Speech] Presentation voice optimized for live demo.", flush=True)

        with self._state_lock:
            self._queued_text += 1
            if not self._speech_active:
                self._speech_active = True
                self.event_bus.emit("speech_started")

        self._queue.put(queued_text)

    def _get_coalesced_text(self, first_text):
        """Combine chunks already arriving, with a tiny debounce window."""
        parts = [first_text]
        deadline = time.monotonic() + self.coalesce_window

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_text = self._queue.get(timeout=remaining)
            except queue.Empty:
                break

            if next_text is None:
                self._queue.task_done()
                self._queue.put(None)
                break

            parts.append(next_text)
            with self._state_lock:
                self._queued_text = max(0, self._queued_text - 1)
            self._queue.task_done()

        return " ".join(part.strip() for part in parts if part and part.strip())

    def _synthesis_loop(self):
        while self._running:
            try:
                text = self._queue.get()
                if text is None:
                    self._queue.task_done()
                    break

                with self._state_lock:
                    self._queued_text = max(0, self._queued_text - 1)

                text = self._get_coalesced_text(text)
                if not text:
                    self._queue.task_done()
                    continue

                with self._state_lock:
                    self._synthesis_inflight += 1

                print(f"[Speech] Synthesizing: {text}", flush=True)
                try:
                    audio = self.engine.synthesize(text)
                    if audio is not None:
                        with self._state_lock:
                            self._pending_audio += 1
                        self._audio_queue.put(audio)
                except Exception as exc:
                    print(
                        f"[Speech] Synthesis worker error: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    with self._state_lock:
                        self._synthesis_inflight -= 1
                    self._queue.task_done()
                    self._maybe_finish_speech()
            except Exception as exc:
                print(
                    f"[Speech] Worker error: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _maybe_finish_speech(self):
        with self._state_lock:
            if not self._speech_active:
                return False
            if self._queued_text != 0 or self._pending_audio != 0 or self._synthesis_inflight != 0:
                return False
            self._speech_active = False

        self.event_bus.emit("speech_finished")
        return True

    def _playback_loop(self):
        while self._running:
            try:
                audio = self._audio_queue.get()
                if audio is None:
                    self._audio_queue.task_done()
                    break

                try:
                    self.engine.play(audio)
                except Exception as exc:
                    print(
                        f"[Speech] Playback worker error: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    with self._state_lock:
                        self._pending_audio = max(0, self._pending_audio - 1)
                    self._audio_queue.task_done()
                    self._maybe_finish_speech()
            except Exception as exc:
                print(
                    f"[Speech] Playback loop error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
