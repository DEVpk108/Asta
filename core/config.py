from __future__ import annotations

import os
import secrets
from pathlib import Path


# Resolve .env from the repository root, not the current working directory,
# so A.S.T.A. behaves the same no matter where it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
HUD_TOKEN_ENV = "ASTA_HUD_TOKEN"


def _parse_line(line: str) -> tuple[str, str] | None:
    value = str(line).strip()
    if not value or value.startswith("#") or "=" not in value:
        return None

    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        return None

    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"\"", "'"}:
        raw = raw[1:-1]
    return key, raw


def load_local_environment(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Load a small .env-style file without adding a runtime dependency."""
    target = Path(path)
    if not target.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        key, value = parsed
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def update_local_environment(
    values: dict[str, str],
    path: str | Path = DEFAULT_ENV_PATH,
) -> None:
    """Persist non-secret local runtime values into the ignored .env file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    updates = {
        str(key).strip(): str(value).strip()
        for key, value in values.items()
        if str(key).strip()
    }

    lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    written: set[str] = set()
    output: list[str] = []

    for line in lines:
        parsed = _parse_line(line)
        if parsed is None:
            output.append(line)
            continue

        key, _ = parsed
        if key in updates:
            output.append(f"{key}={updates[key]}")
            written.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in written:
            if output and output[-1].strip():
                output.append("")
            output.append(f"{key}={value}")

    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    temp.replace(target)

    for key, value in updates.items():
        os.environ[key] = value


def ensure_hud_token() -> str:
    """Return the per-run HUD transport secret, generating one if needed.

    The token is placed in the process environment so child processes (the
    Electron HUD) inherit it. It is never written to disk.
    """
    token = os.environ.get(HUD_TOKEN_ENV, "").strip()
    if not token:
        token = secrets.token_urlsafe(32)
        os.environ[HUD_TOKEN_ENV] = token
    return token
