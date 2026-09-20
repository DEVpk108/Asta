from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class AuthorityStore:
    """Persist authority rules as a small versioned JSON document."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str] | None = None):
        configured = str(path).strip() if path is not None else ""
        if not configured:
            configured = os.getenv("ASTA_AUTHORITY_PATH", "").strip()

        if configured:
            self.path = Path(configured).expanduser().resolve()
        else:
            self.path = Path.home() / ".asta" / "authority.json"

    def load(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()

        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ()

        if not isinstance(payload, dict):
            return ()
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            return ()

        rules = payload.get("rules")
        if not isinstance(rules, list):
            return ()

        valid: list[dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue

            tool = str(rule.get("tool", "")).strip()
            mode = str(rule.get("mode", "")).strip().lower()
            reason = str(rule.get("reason", "")).strip()

            if not tool or mode not in {"auto", "confirm", "deny"}:
                continue

            valid.append(
                {
                    "tool": tool,
                    "mode": mode,
                    "reason": reason,
                }
            )

        return tuple(valid)

    def save(self, rules: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "rules": [dict(rule) for rule in rules],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_name, self.path)

            if os.name != "nt":
                try:
                    self.path.chmod(0o600)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
