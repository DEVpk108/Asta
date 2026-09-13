import queue
import threading
import tkinter as tk
from tkinter import ttk


class HUDApp:
    """Small dependency-free desktop HUD for A.S.T.A. Phase 1."""

    BG = "#08111f"
    PANEL = "#0d1a2b"
    PANEL_2 = "#10233a"
    FG = "#e8f1ff"
    MUTED = "#8ca3bf"
    ACCENT = "#3aa0ff"
    SUCCESS = "#42d392"
    WARNING = "#f5c451"
    ERROR = "#ff6678"
    USER = "#86bfff"

    def __init__(self, event_queue: queue.Queue):
        self._queue = event_queue
        self._thread = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._root = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run,
            name="ASTAHUD",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def stop(self):
        self._queue.put(("shutdown", {}))
        self._closed.wait(timeout=2.0)
        self._thread = None

    def _run(self):
        try:
            root = tk.Tk()
            self._root = root
            root.title("A.S.T.A. — Cognitive HUD")
            root.geometry("1120x720")
            root.minsize(900, 600)
            root.configure(bg=self.BG)
            root.attributes("-topmost", True)

            self._build(root)
            self._ready.set()
            root.after(50, self._poll)
            root.protocol("WM_DELETE_WINDOW", self._close_from_ui)
            root.mainloop()
        except Exception as exc:
            print(f"[HUD] UI error: {type(exc).__name__}: {exc}", flush=True)
            self._ready.set()
        finally:
            self._closed.set()

    def _build(self, root):
        header = tk.Frame(root, bg=self.BG)
        header.pack(fill="x", padx=22, pady=(18, 10))

        title = tk.Label(
            header,
            text="A.S.T.A.",
            bg=self.BG,
            fg=self.FG,
            font=("Segoe UI Semibold", 28),
        )
        title.pack(side="left")

        self.state_label = tk.Label(
            header,
            text="IDLE",
            bg=self.PANEL_2,
            fg=self.MUTED,
            font=("Segoe UI Semibold", 11),
            padx=14,
            pady=6,
        )
        self.state_label.pack(side="right")

        mode = tk.Label(
            header,
            text="LOCAL-FIRST • COGNITIVE HUD",
            bg=self.BG,
            fg=self.ACCENT,
            font=("Segoe UI", 10),
        )
        mode.pack(side="right", padx=(0, 14))

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=22, pady=8)

        conversation_panel = tk.Frame(body, bg=self.PANEL)
        conversation_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

        activity_panel = tk.Frame(body, bg=self.PANEL)
        activity_panel.pack(side="right", fill="both", expand=False, padx=(8, 0))
        activity_panel.configure(width=350)
        activity_panel.pack_propagate(False)

        tk.Label(
            conversation_panel,
            text="CONVERSATION",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(14, 7))

        text_frame = tk.Frame(conversation_panel, bg=self.PANEL)
        text_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.chat = tk.Text(
            text_frame,
            bg=self.BG,
            fg=self.FG,
            insertbackground=self.FG,
            relief="flat",
            wrap="word",
            font=("Cascadia Code", 11),
            padx=14,
            pady=14,
        )
        self.chat.pack(side="left", fill="both", expand=True)
        self.chat.configure(state="disabled")

        self.chat.tag_configure("user", foreground=self.USER, spacing1=12)
        self.chat.tag_configure("assistant", foreground=self.FG, spacing1=12)
        self.chat.tag_configure("meta", foreground=self.MUTED, spacing1=8)
        self.chat.tag_configure("error", foreground=self.ERROR, spacing1=8)

        tk.Label(
            activity_panel,
            text="ACTIVITY",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(14, 7))

        self.activity = tk.Text(
            activity_panel,
            bg=self.BG,
            fg=self.FG,
            relief="flat",
            wrap="word",
            font=("Cascadia Code", 10),
            padx=12,
            pady=12,
        )
        self.activity.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.activity.configure(state="disabled")
        self.activity.tag_configure("ok", foreground=self.SUCCESS)
        self.activity.tag_configure("warn", foreground=self.WARNING)
        self.activity.tag_configure("err", foreground=self.ERROR)
        self.activity.tag_configure("muted", foreground=self.MUTED)

        footer = tk.Frame(root, bg=self.BG)
        footer.pack(fill="x", padx=22, pady=(2, 18))

        self.status_vars = {}
        for key, label in (
            ("VOICE", "VOICE"),
            ("AI", "AI"),
            ("TOOLS", "TOOLS"),
            ("MODE", "MODE"),
        ):
            var = tk.StringVar(value=f"{label} • READY")
            self.status_vars[key] = var
            tk.Label(
                footer,
                textvariable=var,
                bg=self.PANEL,
                fg=self.MUTED,
                font=("Segoe UI Semibold", 9),
                padx=12,
                pady=8,
            ).pack(side="left", padx=(0, 8))

    def _poll(self):
        while True:
            try:
                event, payload = self._queue.get_nowait()
            except queue.Empty:
                break

            if event == "shutdown":
                self._close_from_ui()
                return
            self._handle(event, payload)

        if self._root and not self._closed.is_set():
            self._root.after(50, self._poll)

    def _handle(self, event, payload):
        if event == "user_message":
            text = payload.get("text", "")
            self._set_state("PROCESSING")
            self._append_chat("YOU", text, "user")
            self.status_vars["AI"].set("AI • THINKING")
            return

        if event == "assistant_response":
            text = payload.get("text", "")
            self._set_state("SPEAKING")
            self._append_chat("ASTA", text, "assistant")
            self.status_vars["AI"].set("AI • READY")
            return

        if event == "speech_started":
            self._set_state("SPEAKING")
            self.status_vars["VOICE"].set("VOICE • SPEAKING")
            return

        if event == "speech_finished":
            self.status_vars["VOICE"].set("VOICE • READY")
            if self.state_label.cget("text") == "SPEAKING":
                self._set_state("IDLE")
            return

        if event == "speech_interrupt":
            self._set_state("PROCESSING")
            self.status_vars["VOICE"].set("VOICE • INTERRUPTED")
            self._append_activity("Barge-in / speech interrupted", "warn")
            return

        if event == "tool_request":
            request = payload.get("request")
            name = getattr(request, "tool", "tool")
            self._set_state("EXECUTING TOOL")
            self.status_vars["TOOLS"].set("TOOLS • RUNNING")
            self._append_activity(f"→ {name}", "muted")
            return

        if event == "tool_result":
            result = payload.get("result")
            name = getattr(result, "tool", "tool")
            success = bool(getattr(result, "success", False))
            message = getattr(result, "output", None) or getattr(result, "error", None)
            tag = "ok" if success else "err"
            prefix = "✓" if success else "✕"
            self._append_activity(f"{prefix} {name}: {message}", tag)
            self.status_vars["TOOLS"].set("TOOLS • READY")
            self._set_state("RESULT" if success else "ERROR")
            return

        if event == "conversation_mode_set":
            enabled = bool(payload.get("enabled"))
            self.status_vars["MODE"].set(
                f"MODE • {'CONVERSATION' if enabled else 'WAKEWORD'}"
            )
            self._append_activity(
                f"Conversation mode {'enabled' if enabled else 'disabled'}",
                "ok" if enabled else "muted",
            )
            return

        if event == "hud_rendered":
            self._set_state("SPEAKING")
            return

    def _set_state(self, state):
        self.state_label.configure(text=state)
        color = self.MUTED
        if state in {"SPEAKING", "EXECUTING TOOL"}:
            color = self.ACCENT
        elif state == "RESULT":
            color = self.SUCCESS
        elif state == "ERROR":
            color = self.ERROR
        elif state == "PROCESSING":
            color = self.WARNING
        self.state_label.configure(fg=color)

    def _append_chat(self, speaker, text, tag):
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}\n", "meta")
        self.chat.insert("end", f"{text}\n\n", tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _append_activity(self, text, tag="muted"):
        self.activity.configure(state="normal")
        self.activity.insert("end", f"{text}\n", tag)
        self.activity.see("end")
        self.activity.configure(state="disabled")

    def _close_from_ui(self):
        if self._root:
            self._root.destroy()
