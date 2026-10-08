"""Persist a small, privacy-conscious workspace identity snapshot."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_SCHEMA_VERSION = 1
_MAX_STATE_BYTES = 64 * 1024
_MAX_FILE_PATHS = 20
_MAX_TEXT_LENGTHS = {
    "project_name": 256,
    "project_path": 4096,
    "repository": 2048,
    "branch": 256,
}
_IGNORED_PATH_PARTS = {
    ".git",
    ".ssh",
    ".aws",
    ".azure",
    ".venv",
    "venv",
    "node_modules",
    "credentials",
    "secrets",
}
_SECRET_NAME = re.compile(r"(?:^|[._-])(?:secrets?|credentials?|tokens?)(?:[._-]|$)", re.I)


class WorkspaceStateStore:
    """Atomically store only project identity and safe relative file paths."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        configured = str(path).strip() if path is not None else ""
        if not configured:
            configured = os.getenv("ASTA_WORKSPACE_STATE_PATH", "").strip()
        self.path = (
            Path(configured).expanduser().resolve()
            if configured
            else Path.home() / ".asta" / "workspace.json"
        )

    def load(self) -> dict[str, Any]:
        try:
            if not self.path.is_file() or self.path.stat().st_size > _MAX_STATE_BYTES:
                return {}
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}

        if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
            return {}
        try:
            return self._clean(payload)
        except (OSError, RuntimeError, TypeError, ValueError):
            # A stale or hand-edited state file must never block startup.
            return {}

    def save(self, workspace: dict[str, Any]) -> None:
        if not isinstance(workspace, dict):
            return
        payload = {
            "schema_version": _SCHEMA_VERSION,
            **self._clean(workspace),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        if len(encoded.encode("utf-8")) > _MAX_STATE_BYTES:
            raise ValueError("Workspace state exceeds its storage limit.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if os.name != "nt":
                try:
                    self.path.chmod(0o600)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @classmethod
    def _clean(cls, value: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, max_length in _MAX_TEXT_LENGTHS.items():
            text = value.get(key)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if "\x00" in text:
                continue
            if len(text) > max_length:
                text = text[:max_length]
            if key == "project_path" and text and not cls._is_absolute_path(text):
                continue
            if key == "repository":
                text = cls._sanitize_repository(text)
            cleaned[key] = text

        for key in ("active_files", "recent_files"):
            cleaned[key] = cls._safe_file_paths(value.get(key))
        return cleaned

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        # State written by ASTA contains a resolved path. Avoid expanduser()
        # here: malformed "~user" values can raise when that account is absent.
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()

    @staticmethod
    def _safe_file_paths(value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        result = []
        for raw in value:
            if not isinstance(raw, str):
                continue
            normalized = raw.strip().replace("\\", "/")
            if not normalized or "\x00" in normalized or normalized.startswith("/"):
                continue
            windows = PureWindowsPath(raw.strip())
            if windows.is_absolute() or windows.drive:
                continue
            parts = [part for part in normalized.split("/") if part not in {"", "."}]
            if not parts or any(part == ".." for part in parts):
                continue
            lower_parts = [part.lower() for part in parts]
            if any(part in _IGNORED_PATH_PARTS for part in lower_parts):
                continue
            if any(
                part == ".env"
                or (part.startswith(".env.") and part != ".env.example")
                or Path(part).suffix.lower() in {".key", ".pem", ".p12", ".pfx"}
                or _SECRET_NAME.search(part)
                for part in lower_parts
            ):
                continue
            path = "/".join(parts)
            if len(path) > 512 or path in result:
                continue
            result.append(path)
            if len(result) >= _MAX_FILE_PATHS:
                break
        return result

    @staticmethod
    def _sanitize_repository(value: str) -> str:
        if not value:
            return value
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            hostname = parsed.hostname or parsed.netloc
            netloc = hostname
            try:
                if parsed.port:
                    netloc = f"{hostname}:{parsed.port}"
            except ValueError:
                return ""
            # Remote URLs can contain credentials, access tokens, or query
            # parameters. Preserve only the source host and repository path.
            return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        if "@" in value:
            host, separator, repository_path = value.partition(":")
            if separator:
                return f"{host.rsplit('@', 1)[-1]}:{repository_path}"
        return value[: _MAX_TEXT_LENGTHS["repository"]]