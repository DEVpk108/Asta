from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

_CLEAN = re.compile(r"[^a-z0-9]+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ApplicationRecord:
    name: str
    launch_target: str
    provider: str
    app_id: str | None = None
    source: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_application_name(self.name)


@dataclass(frozen=True, slots=True)
class RunningProcessRecord:
    pid: int
    name: str
    executable_path: str | None = None
    window_title: str | None = None


class ApplicationResolutionError(RuntimeError):
    pass


class ApplicationManager:
    """Generic Windows application discovery; no per-app aliases."""

    def __init__(self, *, refresh_interval_seconds=300.0, powershell_runner=None):
        self.refresh_interval_seconds = max(0.0, float(refresh_interval_seconds))
        self._powershell_runner = powershell_runner or _run_powershell
        self._applications: tuple[ApplicationRecord, ...] = ()
        self._last_refresh = 0.0
        self._last_error: str | None = None
        self._last_opened_application: ApplicationRecord | None = None

    def refresh(self):
        if os.name != "nt":
            self._applications = ()
            self._last_refresh = time.monotonic()
            return self._applications

        records: list[ApplicationRecord] = []
        try:
            records.extend(self._discover_start_apps())
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"

        records.extend(self._discover_start_menu())
        self._applications = _dedupe(records)
        self._last_refresh = time.monotonic()
        return self._applications

    def applications(self, *, refresh=False):
        if refresh or not self._applications or time.monotonic() - self._last_refresh >= self.refresh_interval_seconds:
            return self.refresh()
        return self._applications

    def discover(self, query=None, *, limit=8, refresh=False):
        apps = self.applications(refresh=refresh)
        if not query or not str(query).strip():
            return apps[:limit]

        ranked = []
        for app in apps:
            score = _score(str(query), app.name)
            if score:
                ranked.append((score, app))
        ranked.sort(key=lambda x: (-x[0], x[1].name.lower()))
        return tuple(app for _, app in ranked[: max(1, int(limit))])

    def resolve(self, query: str, *, refresh=False) -> ApplicationRecord:
        query = str(query).strip()
        if not query:
            raise ApplicationResolutionError("Application name cannot be empty.")

        matches = self.discover(query, limit=6, refresh=refresh)
        if not matches and not refresh:
            matches = self.discover(query, limit=6, refresh=True)
        if not matches:
            raise ApplicationResolutionError(
                f"No installed application matched '{query}'."
            )

        top = _score(query, matches[0].name)
        if top < 0.55:
            raise ApplicationResolutionError(
                f"No installed application matched '{query}' confidently."
            )

        if len(matches) > 1:
            second = _score(query, matches[1].name)
            if second >= 0.90 and top - second < 0.05:
                names = ", ".join(app.name for app in matches[:4])
                raise ApplicationResolutionError(
                    f"Application query '{query}' is ambiguous. Candidates: {names}"
                )

        return matches[0]

    def discover_running_processes(self, query=None, *, limit=8):
        """Discover currently running Windows processes and optionally rank them by query."""
        if os.name != "nt":
            return ()

        processes = self._discover_windows_processes()
        if not query or not str(query).strip():
            return tuple(processes[: max(1, int(limit))])

        query = self.resolve_reference(str(query).strip())
        comparison_names = [query]
        try:
            comparison_names.append(self.resolve(query).name)
        except ApplicationResolutionError:
            pass

        ranked = []
        for process in processes:
            fields = (
                process.name or "",
                process.executable_path or "",
                process.window_title or "",
            )
            score = max(
                _score(candidate, field)
                for candidate in comparison_names
                for field in fields
                if field
            )
            if score:
                ranked.append((score, process))

        ranked.sort(key=lambda item: (-item[0], item[1].pid))
        return tuple(process for _, process in ranked[: max(1, int(limit))])

    def resolve_running_process(self, query: str) -> RunningProcessRecord:
        query = self.resolve_reference(str(query).strip())
        if not query:
            raise ApplicationResolutionError("Application name cannot be empty.")

        matches = self.discover_running_processes(query, limit=6)
        if not matches:
            raise ApplicationResolutionError(
                f"No running application matched '{query}'."
            )

        comparison_names = [query]
        try:
            comparison_names.append(self.resolve(query).name)
        except ApplicationResolutionError:
            pass

        top = max(
            (
                _score_running_process(candidate, matches[0])
                for candidate in comparison_names
            ),
            default=0.0,
        )
        if top < 0.80:
            raise ApplicationResolutionError(
                f"No running application matched '{query}' confidently."
            )

        return matches[0]

    def _discover_windows_processes(self):
        output = self._powershell_runner(
            "Get-Process | ForEach-Object { "
            "$path = $null; "
            "try { $path = $_.Path } catch {} "
            "[pscustomobject]@{ "
            "ProcessId = [int]$_.Id; "
            "Name = [string]$_.ProcessName; "
            "Path = [string]$path; "
            "WindowTitle = [string]$_.MainWindowTitle "
            "} "
            "} | ConvertTo-Json -Compress"
        )
        if not output.strip():
            return []

        payload = json.loads(output)
        entries = payload if isinstance(payload, list) else [payload]
        records = []

        for item in entries:
            if not isinstance(item, dict):
                continue
            try:
                pid = int(item.get("ProcessId"))
            except (TypeError, ValueError):
                continue
            if pid <= 0:
                continue

            name = str(item.get("Name") or "").strip()
            path = str(item.get("Path") or "").strip() or None
            window_title = str(item.get("WindowTitle") or "").strip() or None
            if not name:
                continue

            records.append(
                RunningProcessRecord(
                    pid=pid,
                    name=name,
                    executable_path=path,
                    window_title=window_title,
                )
            )
        return records


    @property
    def last_error(self):
        return self._last_error

    @property
    def last_opened_application(self) -> ApplicationRecord | None:
        return self._last_opened_application

    def remember_opened(self, application: ApplicationRecord) -> None:
        if not isinstance(application, ApplicationRecord):
            raise TypeError("application must be an ApplicationRecord")
        self._last_opened_application = application

    def resolve_reference(self, query: str) -> str:
        value = str(query).strip()
        normalized = normalize_application_name(value)
        if normalized in {"it", "this", "that", "the app", "the application"}:
            application = self._last_opened_application
            if application is None:
                raise ApplicationResolutionError(
                    f"No recent application reference is available for '{value}'."
                )
            return application.name
        return value

    def _discover_start_apps(self):
        output = self._powershell_runner(
            "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
        )
        if not output.strip():
            return []

        payload = json.loads(output)
        entries = payload if isinstance(payload, list) else [payload]
        records = []

        for item in entries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name") or "").strip()
            app_id = str(item.get("AppID") or "").strip()
            if not name or not app_id:
                continue
            records.append(
                ApplicationRecord(
                    name=name,
                    launch_target=f"shell:AppsFolder\\{app_id}",
                    provider="windows.start_apps",
                    app_id=app_id,
                    source="Get-StartApps",
                )
            )
        return records

    def _discover_start_menu(self):
        roots = (
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        )
        records = []
        seen = set()

        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_dir() or path.suffix.lower() not in {".lnk", ".url", ".exe"}:
                    continue
                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)
                if path.stem.strip():
                    records.append(
                        ApplicationRecord(
                            name=path.stem.strip(),
                            launch_target=str(path),
                            provider="windows.start_menu",
                            source="Start Menu",
                        )
                    )
        return records


def normalize_application_name(value: str) -> str:
    text = str(value).strip().lower().replace("&", " and ")
    return _SPACES.sub(" ", _CLEAN.sub(" ", text)).strip()


def _score(query: str, candidate: str) -> float:
    q = normalize_application_name(query)
    c = normalize_application_name(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c:
        return 0.94
    qt = q.split()
    ct = c.split()
    if all(token in ct for token in qt):
        return 0.92
    if _ordered_token_prefix_match(qt, ct):
        return 0.91
    if _abbreviation_match(qt, ct):
        return 0.90
    compact_q = "".join(qt)
    compact_c = "".join(ct)
    if compact_q in compact_c:
        return 0.88
    return SequenceMatcher(None, compact_q, compact_c, autojunk=False).ratio() * 0.84


def _ordered_token_prefix_match(query_tokens, candidate_tokens):
    position = 0
    for query_token in query_tokens:
        while position < len(candidate_tokens):
            candidate_token = candidate_tokens[position]
            position += 1
            if candidate_token.startswith(query_token):
                break
        else:
            return False
    return True


def _abbreviation_match(query_tokens, candidate_tokens):
    """Match abbreviated query tokens against candidate token prefixes/initials."""
    if not query_tokens or not candidate_tokens:
        return False

    candidate_positions = 0
    for query_token in query_tokens:
        matched = False

        while candidate_positions < len(candidate_tokens):
            candidate_token = candidate_tokens[candidate_positions]

            if candidate_token.startswith(query_token):
                candidate_positions += 1
                matched = True
                break

            # Try compact forms such as "vscode":
            # "v" from "visual" + "s" from "studio" + "code" from "code".
            remaining = query_token
            lookahead = candidate_positions
            while lookahead < len(candidate_tokens) and remaining:
                token = candidate_tokens[lookahead]

                if token.startswith(remaining):
                    remaining = ""
                    lookahead += 1
                    break

                prefix = token[0]
                if not remaining.startswith(prefix):
                    break

                remaining = remaining[1:]
                lookahead += 1

            if not remaining:
                candidate_positions = lookahead
                matched = True
                break

            candidate_positions += 1

        if not matched:
            return False

    return True


def _score_running_process(query: str, process: RunningProcessRecord) -> float:
    fields = (
        process.name or "",
        process.executable_path or "",
        process.window_title or "",
    )
    return max((_score(query, field) for field in fields if field), default=0.0)


def _dedupe(records):
    result = []
    seen_names = set()
    seen_targets = set()

    for app in records:
        key = app.normalized_name
        target = app.launch_target.lower()
        if (key, target) in seen_targets:
            continue
        seen_targets.add((key, target))

        if key in seen_names:
            existing = next(x for x in result if x.normalized_name == key)
            if existing.provider == "windows.start_apps" and app.provider != "windows.start_apps":
                continue
            result.remove(existing)

        seen_names.add(key)
        result.append(app)

    result.sort(key=lambda x: x.name.lower())
    return tuple(result)


def _run_powershell(command: str) -> str:
    executable = (
        shutil.which("powershell.exe")
        or shutil.which("powershell")
        or shutil.which("pwsh.exe")
        or shutil.which("pwsh")
    )
    if executable is None:
        raise RuntimeError("PowerShell was not found.")

    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=5,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "unknown PowerShell error")
    return result.stdout
