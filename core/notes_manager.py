from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class NotesManager:
    """Persist user-authored notes as local Markdown files.

    Notes are intentionally separate from semantic memory: a note is a
    user-visible artifact that can be listed, read, searched, and edited later.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None):
        configured = root or os.getenv("ASTA_NOTES_PATH", "").strip()
        self.root = Path(configured).expanduser() if configured else (
            Path(__file__).resolve().parents[1] / "data" / "notes"
        )
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        content: str,
        *,
        title: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        body = self._validate_content(content)
        note_title = self._normalize_title(title) if title else self._derive_title(body)
        path = self.root / f"{self._slug(note_title)}.md"

        if path.exists() and not overwrite:
            raise FileExistsError(f"Note already exists: {note_title}")

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        markdown = (
            f"# {note_title}\n\n"
            f"{body.strip()}\n\n"
            f"<!-- asta-note-created: {now} -->\n"
        )
        path.write_text(markdown, encoding="utf-8")

        return {
            "title": note_title,
            "path": str(path),
            "created": True,
        }

    def read(self, title: str) -> dict[str, Any]:
        note = self._resolve(title)
        text = note.read_text(encoding="utf-8")
        note_title, body = self._parse(text, fallback=note.stem)
        return {
            "title": note_title,
            "path": str(note),
            "content": body,
        }

    def list(self, *, limit: int = 50) -> list[dict[str, str]]:
        limit = max(1, int(limit))
        notes = []
        for path in sorted(self.root.glob("*.md"), key=lambda item: item.name.lower()):
            try:
                text = path.read_text(encoding="utf-8")
                title, _ = self._parse(text, fallback=path.stem)
            except OSError:
                continue
            notes.append({"title": title, "path": str(path)})
            if len(notes) >= limit:
                break
        return notes

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        needle = str(query).strip().lower()
        if not needle:
            raise ValueError("Search query must be non-empty.")

        results = []
        for item in self.list(limit=1000):
            try:
                text = Path(item["path"]).read_text(encoding="utf-8")
            except OSError:
                continue

            if needle not in text.lower():
                continue

            _, body = self._parse(text, fallback=item["title"])
            results.append(
                {
                    "title": item["title"],
                    "path": item["path"],
                    "content": body,
                }
            )
            if len(results) >= max(1, int(limit)):
                break

        return results

    def append(self, title: str, content: str) -> dict[str, Any]:
        body = self._validate_content(content)
        note = self._resolve(title)
        existing = note.read_text(encoding="utf-8").rstrip()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        note.write_text(
            f"{existing}\n\n{body.strip()}\n\n"
            f"<!-- asta-note-updated: {timestamp} -->\n",
            encoding="utf-8",
        )
        note_title, _ = self._parse(note.read_text(encoding="utf-8"), fallback=note.stem)
        return {"title": note_title, "path": str(note), "updated": True}

    def _resolve(self, title: str) -> Path:
        value = str(title).strip()
        if not value:
            raise ValueError("Note title must be non-empty.")

        direct = self.root / f"{self._slug(value)}.md"
        if direct.exists():
            return direct

        normalized = value.casefold()
        for path in self.root.glob("*.md"):
            if path.stem.casefold() == normalized:
                return path

        raise FileNotFoundError(f"Note not found: {value}")

    @staticmethod
    def _validate_content(content: str) -> str:
        value = str(content).strip()
        if not value:
            raise ValueError("Note content must be non-empty.")
        return value

    @staticmethod
    def _normalize_title(title: str) -> str:
        value = " ".join(str(title).strip().split())
        if not value:
            raise ValueError("Note title must be non-empty.")
        return value[:160]

    @classmethod
    def _derive_title(cls, content: str) -> str:
        first_line = content.splitlines()[0].strip()
        first_line = re.sub(r"^[#>*\-\s]+", "", first_line).strip()
        title = first_line[:80].strip(" .,:;!?-")
        if title:
            return cls._normalize_title(title)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H-%M")
        return f"Note {timestamp}"

    @staticmethod
    def _slug(title: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower())
        value = value.strip("-")
        return value or "note"

    @staticmethod
    def _parse(text: str, *, fallback: str) -> tuple[str, str]:
        lines = text.splitlines()
        title = fallback
        start = 0

        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip() or fallback
            start = 1

        body_lines = []
        for line in lines[start:]:
            if line.strip().startswith("<!-- asta-note-"):
                continue
            body_lines.append(line)

        body = "\n".join(body_lines).strip()
        return title, body
