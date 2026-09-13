import queue
import threading
import time

from core.module import Module
from .kokoro_engine import KokoroEngine


class SpeechModule(Module):

    PRESENTATION_SHORT_TEXT = (
        "Hello Sir. I’m A.S.T.A., a local-first AI engineering assistant. "
        "I understand voice commands, reason about technical questions, and use authorized tools to interact with the computer. "
        "My architecture connects voice, AI reasoning, tool execution, approvals, speech, and the HUD through the kernel. "
        "My local AI stack uses LM Studio, Whisper, and Kokoro. "
        "My goal is to grow into a personal AI operating system with stronger memory, workflow awareness, proactive assistance, and specialized agents."
    )

    def __init__(self, kernel):
        super().__init__(
            name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = KokoroEngine()

        # One worker owns synthesis + playback. This prevents later sentences
        # from being synthesized/played over the previous sentence.
        self._queue = queue.Queue()
        self._running = False
        self._speech_thread = None
        self.coalesce_window = 0.08

        self._state_lock = threading.Lock()
        self._speech_active = False
        self._queued_text = 0
        self._synthesis_inflight = 0
        self._presentation_mode_active = False
        self._interrupt_event = threading.Event()

    def initialize(self):
        print("[Speech] Initializing...", flush=True)
        self._running = True
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.subscribe("speech_interrupt", self.on_speech_interrupt)

        self._speech_thread = threading.Thread(
            target=self._speech_loop,
            name="SpeechWorker",
            daemon=True,
        )
        self._speech_thread.start()
        print("[Speech] Ready", flush=True)

    def shutdown(self):
        print("[Speech] Shutting down...", flush=True)
        self._running = False
        self._interrupt_event.set()
        self.event_bus.unsubscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.unsubscribe("speech_interrupt", self.on_speech_interrupt)

        self._queue.put(None)

        if self._speech_thread is not None and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=2.0)

        self._speech_thread = None
        print("[Speech] Stopped", flush=True)

    def on_assistant_sentence(self, text):
        if not text:
            return

        is_presentation = text.startswith("Hello Sir. I’m A.S.T.A.,")
        if is_presentation:
            with self._state_lock:
                self._presentation_mode_active = True
            queued_text = self.PRESENTATION_SHORT_TEXT
            print("[Speech] Presentation voice optimized for live demo.", flush=True)
        elif self._presentation_mode_active:
            return
        else:
            queued_text = text

        with self._state_lock:
            self._queued_text += 1
            was_inactive = not self._speech_active
            self._speech_active = True

        if was_inactive:
            # A new response is allowed to speak after a previous interrupt.
            self._interrupt_event.clear()
            self.event_bus.emit("speech_started")

        self._queue.put(queued_text)

    def on_speech_interrupt(self, *args, **kwargs):
        """Cancel current speech and discard anything queued behind it."""
        print("[Speech] Interrupt requested.", flush=True)
        self._interrupt_event.set()

        drained = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            self._queue.task_done()
            if item is not None:
                drained += 1

        with self._state_lock:
            self._queued_text = 0
            was_active = self._speech_active
            self._speech_active = False
            self._presentation_mode_active = False

        if drained:
            print(f"[Speech] Discarded {drained} queued speech item(s).", flush=True)
        if was_active:
            self.event_bus.emit("speech_finished")

    def _get_coalesced_text(self, first_text):
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

    def _speech_loop(self):
        while self._running:
            try:
                text = self._queue.get()
                if text is None:
                    self._queue.task_done()
                    break

                with self._state_lock:
                    self._queued_text = max(0, self._queued_text - 1)

                # If a new response arrived after an interrupt, this is the
                # first accepted item and the new response should speak normally.
                if self._interrupt_event.is_set():
                    self._interrupt_event.clear()

                text = self._get_coalesced_text(text)
                if not text:
                    self._queue.task_done()
                    self._maybe_finish_speech()
                    continue

                with self._state_lock:
                    self._synthesis_inflight += 1

                print(f"[Speech] Synthesizing: {text}", flush=True)
                try:
                    audio = self.engine.synthesize(text)
                    if (
                        audio is not None
                        and self._running
                        and not self._interrupt_event.is_set()
                    ):
                        completed = self.engine.play(
                            audio,
                            should_continue=lambda: (
                                self._running
                                and not self._interrupt_event.is_set()
                            ),
                        )
                        if not completed:
                            print("[Speech] Playback interrupted.", flush=True)
                except Exception as exc:
                    print(
                        f"[Speech] Speech worker error: {type(exc).__name__}: {exc}",
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
            if self._queued_text != 0 or self._synthesis_inflight != 0:
                return False
            self._speech_active = False
            self._presentation_mode_active = False

        self.event_bus.emit("speech_finished")
        return True
