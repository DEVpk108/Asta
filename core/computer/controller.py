from __future__ import annotations

import math
import threading


class ComputerControlError(RuntimeError):
    """Raised when a local computer-control operation cannot be completed."""


class ComputerController:
    """Small backend-agnostic computer-control facade.

    The default backend is PyAutoGUI, loaded lazily so importing A.S.T.A.'s
    core remains possible even before the optional dependency is installed.
    Tests can inject a fake backend through the constructor.
    """

    def __init__(self, backend=None, clipboard=None):
        self._backend = backend
        self._clipboard = clipboard
        self._lock = threading.RLock()

    def _load_backend(self):
        if self._backend is not None:
            return self._backend

        try:
            import pyautogui
        except ImportError as exc:
            raise ComputerControlError(
                "Computer control requires the 'pyautogui' package."
            ) from exc

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.03
        self._backend = pyautogui
        return self._backend

    @staticmethod
    def _coordinate(value, name):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ComputerControlError(
                f"{name} must be a finite number."
            ) from exc
        if not math.isfinite(number):
            raise ComputerControlError(f"{name} must be a finite number.")
        return int(round(number))

    @staticmethod
    def _duration(value):
        try:
            duration = float(value)
        except (TypeError, ValueError) as exc:
            raise ComputerControlError(
                "duration must be a non-negative number."
            ) from exc
        if not math.isfinite(duration) or duration < 0:
            raise ComputerControlError(
                "duration must be a non-negative finite number."
            )
        return duration

    def move_mouse(self, *, x, y, duration=0.0):
        backend = self._load_backend()
        x = self._coordinate(x, "x")
        y = self._coordinate(y, "y")
        duration = self._duration(duration)
        with self._lock:
            backend.moveTo(x, y, duration=duration)
        return {"x": x, "y": y, "moved": True}

    def click(self, *, x=None, y=None, button="left", clicks=1, interval=0.0):
        backend = self._load_backend()
        if x is not None and y is not None:
            x = self._coordinate(x, "x")
            y = self._coordinate(y, "y")
        elif x is not None or y is not None:
            raise ComputerControlError("x and y must be supplied together.")

        button = str(button or "left").strip().lower()
        if button not in {"left", "middle", "right"}:
            raise ComputerControlError(
                "button must be one of: left, middle, right."
            )

        try:
            clicks = int(clicks)
            interval = float(interval)
        except (TypeError, ValueError) as exc:
            raise ComputerControlError(
                "clicks must be an integer and interval must be a number."
            ) from exc

        if clicks < 1:
            raise ComputerControlError("clicks must be at least 1.")
        if not math.isfinite(interval) or interval < 0:
            raise ComputerControlError(
                "interval must be a non-negative finite number."
            )

        with self._lock:
            backend.click(
                x=x,
                y=y,
                button=button,
                clicks=clicks,
                interval=interval,
            )

        output = {
            "button": button,
            "clicks": clicks,
            "clicked": True,
        }
        if x is not None:
            output["x"] = x
            output["y"] = y
        return output

    def type_text(self, text, *, interval=0.0, paste=True):
        value = str(text or "")
        if not value:
            raise ComputerControlError("text must be non-empty.")

        backend = self._load_backend()
        try:
            interval = float(interval)
        except (TypeError, ValueError) as exc:
            raise ComputerControlError(
                "interval must be a non-negative number."
            ) from exc
        if not math.isfinite(interval) or interval < 0:
            raise ComputerControlError(
                "interval must be a non-negative finite number."
            )

        with self._lock:
            if paste:
                clipboard = self._clipboard
                if clipboard is None:
                    try:
                        import pyperclip
                    except ImportError:
                        pyperclip = None
                    clipboard = pyperclip

                if clipboard is not None:
                    try:
                        previous = clipboard.paste()
                        clipboard.copy(value)
                        backend.hotkey("ctrl", "v")
                        clipboard.copy(previous)
                        return {
                            "text_length": len(value),
                            "typed": True,
                            "method": "clipboard_paste",
                        }
                    except Exception:
                        # Fall through to direct typing for environments where
                        # clipboard access is restricted.
                        pass

            try:
                backend.write(value, interval=interval)
            except Exception as exc:
                raise ComputerControlError(
                    f"Direct text entry failed: {type(exc).__name__}: {exc}"
                ) from exc

        return {
            "text_length": len(value),
            "typed": True,
            "method": "direct_typing",
        }

    def keypress(self, key):
        key = str(key or "").strip().lower()
        if not key:
            raise ComputerControlError("key must be non-empty.")

        backend = self._load_backend()
        with self._lock:
            backend.press(key)
        return {"key": key, "pressed": True}

    def hotkey(self, keys):
        if isinstance(keys, str):
            keys = [item.strip() for item in keys.split("+") if item.strip()]
        if not isinstance(keys, (list, tuple)):
            raise ComputerControlError("keys must be a list or '+'-separated string.")

        normalized = [
            str(key).strip().lower()
            for key in keys
            if str(key).strip()
        ]
        if not normalized:
            raise ComputerControlError("keys must contain at least one key.")

        backend = self._load_backend()
        with self._lock:
            backend.hotkey(*normalized)
        return {"keys": normalized, "pressed": True}

    @staticmethod
    def _wait_duration(value):
        try:
            duration = float(value)
        except (TypeError, ValueError) as exc:
            raise ComputerControlError(
                "seconds must be a non-negative finite number."
            ) from exc
        if not math.isfinite(duration) or duration < 0:
            raise ComputerControlError(
                "seconds must be a non-negative finite number."
            )
        return min(duration, 10.0)

    def wait(self, *, seconds=0.5):
        duration = self._wait_duration(seconds)
        import time

        time.sleep(duration)
        return {"seconds": duration, "waited": True}

    def scroll(self, amount, *, x=None, y=None):
        backend = self._load_backend()

        try:
            amount = int(amount)
        except (TypeError, ValueError) as exc:
            raise ComputerControlError("amount must be an integer.") from exc

        if x is not None and y is not None:
            x = self._coordinate(x, "x")
            y = self._coordinate(y, "y")
        elif x is not None or y is not None:
            raise ComputerControlError("x and y must be supplied together.")

        with self._lock:
            if x is None:
                backend.scroll(amount)
            else:
                backend.moveTo(x, y)
                backend.scroll(amount)

        output = {"amount": amount, "scrolled": True}
        if x is not None:
            output["x"] = x
            output["y"] = y
        return output
