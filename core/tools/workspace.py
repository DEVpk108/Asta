"""Workspace-scoped text file capabilities.

All paths are relative to the active workspace. These tools intentionally
provide a small, auditable surface for project work: list, read, and
create/update text files. They do not delete or move files.
"""

from __future__ import annotations

import difflib
import hashlib
import hmac
import os
import re
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


MAX_TEXT_FILE_BYTES = 512 * 1024
DEFAULT_LIST_LIMIT = 200
MAX_LIST_LIMIT = 500
_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".ssh",
    ".aws",
    ".azure",
    "credentials",
    "secrets",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}
_SECRET_NAME = re.compile(
    r"(?:^|[._-])(?:secrets?|credentials?|tokens?)(?:[._-]|$)",
    re.I,
)
_SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


class WorkspacePathError(ValueError):
    """A requested path is missing, unsafe, or outside the active workspace."""


def resolve_workspace_root(workspace_manager) -> Path:
    """Return the canonical active workspace root or raise a safe error."""
    if workspace_manager is None:
        raise WorkspacePathError("No workspace manager is configured.")

    state = None
    state_getter = getattr(workspace_manager, "state", None)
    if callable(state_getter):
        state = state_getter()
    elif callable(getattr(workspace_manager, "snapshot", None)):
        state = workspace_manager.snapshot()

    if isinstance(state, dict):
        value = state.get("project_path")
    else:
        value = getattr(state, "project_path", None)

    if not isinstance(value, str) or not value.strip():
        raise WorkspacePathError("No active project workspace is selected.")

    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkspacePathError("The active workspace path is unavailable.") from exc

    if not root.is_dir():
        raise WorkspacePathError("The active workspace is not a directory.")
    return root


def resolve_workspace_path(
    root: Path,
    value: Any,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve a relative path and reject absolute paths and root escapes."""
    if not isinstance(value, str) or not value.strip():
        raise WorkspacePathError("A non-empty workspace-relative path is required.")
    if "\x00" in value:
        raise WorkspacePathError("The path contains an invalid character.")

    normalized = value.strip().replace("\\", "/")
    windows = PureWindowsPath(normalized)
    if normalized.startswith("/") or windows.is_absolute() or windows.drive:
        raise WorkspacePathError("Absolute paths are not allowed; use a workspace-relative path.")

    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise WorkspacePathError("Parent-directory traversal is not allowed.")
    if not parts and not allow_root:
        raise WorkspacePathError("The workspace root is not a file path.")

    candidate = root.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError("The requested path resolves outside the active workspace.") from exc
    return candidate


def existing_workspace_file(workspace_manager, requested_path) -> tuple[str, bytes] | None:
    """Return (sha256, bytes) of an existing, non-sensitive workspace text file."""
    try:
        root = resolve_workspace_root(workspace_manager)
        path = resolve_workspace_path(root, requested_path)
        if _is_sensitive_path(root, path) or _is_sensitive_parts(
            str(requested_path).replace("\\", "/").split("/")
        ):
            return None
        if not path.is_file() or path.stat().st_size > MAX_TEXT_FILE_BYTES:
            return None
        data = path.read_bytes()
    except (OSError, RuntimeError, ValueError, WorkspacePathError):
        return None
    return hashlib.sha256(data).hexdigest(), data


_TEST_PATH = re.compile(
    r"(?:^|/)(?:tests?|__tests__|spec)/|(?:^|/)test_[^/]+$|(?:^|/)[^/]+_test\.[^/]+$"
    r"|(?:^|/)[^/]+\.(?:test|spec)\.[^/]+$|(?:^|/)conftest\.py$",
    re.I,
)
_FILE_MENTION = re.compile(
    r"(?<![\w/\\.-])((?:[\w.-]+[/\\])*[\w-][\w.-]*\.[A-Za-z][A-Za-z0-9]{0,7})(?![\w/\\-])"
)
_SPOKEN_DOT = re.compile(
    r"\b([\w-]+)\s+dot\s+(py|js|ts|tsx|jsx|json|md|txt|toml|yaml|yml|cfg|ini|html|css|sh|ps1)\b",
    re.I,
)
_TEST_WORDS = re.compile(r"\b(?:tests?|pytest|unittest|failing|passes|pass)\b", re.I)
_CONTEXT_SKIP_DIRS = {".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode"}
MAX_CONTEXT_FILE_BYTES = 6000
MAX_CONTEXT_TOTAL_BYTES = 12000
MAX_CONTEXT_FILES = 4
MAX_CONTEXT_LISTING = 60
MAX_DIFF_CHARS = 4000


def is_test_path(path: Any) -> bool:
    """True for files that hold tests (test_x.py, x_test.go, tests/..., x.spec.ts)."""
    return bool(_TEST_PATH.search(str(path or "").replace("\\", "/").lstrip("./")))


def mentioned_paths(text: Any) -> list[str]:
    """File names the user said ("calc.py", "src/app.js", "calc dot py")."""
    value = _SPOKEN_DOT.sub(lambda match: f"{match.group(1)}.{match.group(2).lower()}", str(text or ""))
    found: list[str] = []
    for match in _FILE_MENTION.finditer(value):
        item = match.group(1).replace("\\", "/").rstrip(".")
        if item and item.lower() not in {name.lower() for name in found}:
            found.append(item)
    return found


_CLAUSE_BREAK = re.compile(
    r"\s*(?:[,;]|\.(?=\s|$)|\b(?:so that|so|until|and then|then|and make|to make|making)\b)\s*",
    re.IGNORECASE,
)
_EDIT_WORDS = re.compile(
    r"\b(?:fix|edit|change|update|modify|rewrite|correct|repair|adjust|add|remove|delete|"
    r"rename|refactor|write|replace|tweak|clean up|extend)\b|\b(?:is|are|looks?|seems?)\s+"
    r"(?:wrong|broken|incorrect|outdated|out of date)\b|\btypo\b",
    re.IGNORECASE,
)
_OBJECTIVE_WORDS = re.compile(r"\b(?:pass|passes|passing|green|succeed|succeeds|work|works)\b", re.IGNORECASE)


def edit_requested(goal: Any, path: Any) -> bool:
    """Did the user ask for this file to be changed, not just mention it?

    "Fix the bug in calc.py so the tests in test_calc.py pass" names
    test_calc.py only as the finish line; that is no permission to rewrite
    it. "The assertion in test_calc.py is wrong, fix it" or "update
    test_calc.py" is. Files other than tests count as requested when named.
    """
    target = str(path or "").replace("\\", "/").strip("/").lower()
    if not target:
        return False
    names = {target, target.rsplit("/", 1)[-1]}
    mentioned = {item.lower().strip("/") for item in mentioned_paths(goal)}
    if not (names & mentioned):
        return False
    if not is_test_path(target):
        return True
    value = _SPOKEN_DOT.sub(lambda match: f"{match.group(1)}.{match.group(2).lower()}", str(goal or ""))
    clauses = [part for part in _CLAUSE_BREAK.split(value) if part and part.strip()]
    for index, clause in enumerate(clauses):
        lowered = clause.lower()
        if not any(name in lowered for name in names):
            continue
        if _OBJECTIVE_WORDS.search(clause):
            continue  # "...the tests in test_calc.py pass"
        if _EDIT_WORDS.search(clause):
            return True
        following = clauses[index + 1] if index + 1 < len(clauses) else ""
        if re.match(r"\s*(?:please\s+)?(?:fix|change|update|correct|edit)\s+(?:it|that|this|them)\b", following, re.IGNORECASE):
            return True  # "The assertion in test_calc.py is off, fix it"
    return False


def text_change_summary(before: str | None, after: str, path: str = "") -> dict[str, Any]:
    """Line-level change stats and a bounded unified diff for a text edit."""
    old_lines = (before or "").replace("\r\n", "\n").splitlines(keepends=True)
    new_lines = after.replace("\r\n", "\n").splitlines(keepends=True)
    added = removed = 0
    diff_lines = []
    for line in difflib.unified_diff(old_lines, new_lines, f"a/{path}", f"b/{path}", n=2):
        diff_lines.append(line if line.endswith("\n") else line + "\n")
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    diff = "".join(diff_lines)
    return {
        "lines_added": added,
        "lines_removed": removed,
        "diff": diff[:MAX_DIFF_CHARS],
        "diff_truncated": len(diff) > MAX_DIFF_CHARS,
    }


def planning_file_context(workspace_manager, goal: Any) -> dict[str, Any]:
    """Bounded view of the project for planning an edit.

    Lists a few dozen project files and includes the current text and SHA-256
    of the files the user named (plus their tests), so the planner can write a
    complete, correct update instead of guessing what the file contains.
    Files the user did not name are marked read_only: the plan may read them,
    but changing them needs the user's approval.
    """
    try:
        root = resolve_workspace_root(workspace_manager)
    except WorkspacePathError:
        return {}

    listing: list[str] = []
    try:
        for current, directories, files in os.walk(root):
            base = Path(current)
            relative_base = base.relative_to(root)
            depth = len(relative_base.parts)
            directories[:] = sorted(
                name for name in directories
                if depth < 2
                and not name.startswith(".")
                and name.lower() not in _CONTEXT_SKIP_DIRS
                and not _is_sensitive_parts((*relative_base.parts, name))
            )
            for name in sorted(files):
                parts = (*relative_base.parts, name)
                if name.startswith(".asta-write-") or _is_sensitive_parts(parts):
                    continue
                listing.append("/".join(parts))
                if len(listing) >= MAX_CONTEXT_LISTING:
                    break
            if len(listing) >= MAX_CONTEXT_LISTING:
                break
    except OSError:
        pass

    by_name: dict[str, list[str]] = {}
    for item in listing:
        by_name.setdefault(item.rsplit("/", 1)[-1].lower(), []).append(item)
    lowered = {item.lower(): item for item in listing}

    named: list[str] = []
    for mention in mentioned_paths(goal):
        exact = lowered.get(mention.lower())
        matches = [exact] if exact else by_name.get(mention.rsplit("/", 1)[-1].lower(), [])
        if len(matches) == 1 and matches[0] not in named:
            named.append(matches[0])

    # A test file named only as the goal ("so test_calc.py passes") is read
    # only; the user has to ask for it to change.
    selected: list[tuple[str, bool]] = [(item, edit_requested(goal, item)) for item in named]
    if _TEST_WORDS.search(str(goal or "")) or named:
        for item in named:
            stem = Path(item).stem.lower()
            for candidate in listing:
                name = candidate.rsplit("/", 1)[-1].lower()
                if (
                    candidate not in named
                    and is_test_path(candidate)
                    and stem
                    and stem in {name.split(".")[0].removeprefix("test_"), name.split(".")[0].removesuffix("_test")}
                ):
                    selected.append((candidate, False))
        if not named and _TEST_WORDS.search(str(goal or "")):
            tests = [item for item in listing if is_test_path(item) and item.endswith(".py")]
            if len(tests) == 1:
                selected.append((tests[0], False))

    contents: list[dict[str, Any]] = []
    total = 0
    seen = set()
    for item, editable in selected:
        if item in seen or len(contents) >= MAX_CONTEXT_FILES:
            continue
        seen.add(item)
        try:
            path = resolve_workspace_path(root, item)
            if _is_sensitive_path(root, path) or not path.is_file():
                continue
            raw = path.read_bytes() if path.stat().st_size <= MAX_TEXT_FILE_BYTES else None
        except (OSError, RuntimeError, ValueError, WorkspacePathError):
            continue
        entry: dict[str, Any] = {"path": item, "read_only": not editable}
        if raw is None or len(raw) > MAX_CONTEXT_FILE_BYTES or total + len(raw) > MAX_CONTEXT_TOTAL_BYTES:
            entry["too_large_to_include"] = True
        else:
            try:
                entry["content"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            if item.endswith(".py") and is_test_path(item):
                # A script of bare asserts makes pytest report "no tests ran"
                # (exit 5); say how this file is meant to be run.
                has_tests = re.search(r"^\s*(?:async\s+)?def\s+test_|^\s*class\s+Test", entry["content"], re.M)
                entry["runs_with"] = (
                    f"pytest {item}" if has_tests else f"python {item}"
                )
            total += len(raw)
        contents.append(entry)

    if not listing and not contents:
        return {}
    return {
        "files": listing,
        "files_truncated": len(listing) >= MAX_CONTEXT_LISTING,
        "file_contents": contents,
    }


def _is_sensitive_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return _is_sensitive_parts(relative.parts)


def _is_sensitive_parts(parts) -> bool:
    normalized_parts = tuple(str(part).lower() for part in parts)
    if any(part in _IGNORED_DIRECTORY_NAMES for part in normalized_parts):
        return True
    if any(
        part == ".env"
        or part.endswith(".env")
        or (part.startswith(".env.") and part != ".env.example")
        for part in normalized_parts
    ):
        return True

    if not normalized_parts:
        return False
    name = normalized_parts[-1]
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if Path(name).suffix.lower() in _SECRET_SUFFIXES or _SECRET_NAME.search(name):
        return True
    return False


def _tool_result(
    request: ToolRequest,
    *,
    success: bool,
    output=None,
    error: str | None = None,
    start: float,
) -> ToolResult:
    import time

    return ToolResult(
        success=success,
        tool=request.tool,
        output=output,
        error=error,
        duration_seconds=time.perf_counter() - start,
        metadata={"request_id": request.request_id},
    )


class _WorkspaceFileTool(Tool):
    def __init__(self, workspace_manager):
        self.workspace_manager = workspace_manager

    def _root(self) -> Path:
        return resolve_workspace_root(self.workspace_manager)

    def _record_recent_file(self, root: Path, path: Path) -> None:
        recorder = getattr(self.workspace_manager, "add_recent_file", None)
        if not callable(recorder):
            return
        try:
            if resolve_workspace_root(self.workspace_manager) != root:
                return
            recorder(path.relative_to(root).as_posix())
        except (OSError, RuntimeError, ValueError, WorkspacePathError):
            # Recent-file bookkeeping must not turn a successful project read
            # or write into a failed file operation.
            return


class ListWorkspaceFilesTool(_WorkspaceFileTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="filesystem.list_files",
            description=(
                "List files and directories inside the active project workspace. "
                "Paths must be relative to the workspace."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_LIMIT},
                },
                "required": [],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["list_files"],
                "category": "filesystem",
                "workspace_scoped": True,
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        import time

        start = time.perf_counter()
        try:
            root = self._root()
            directory = resolve_workspace_path(
                root,
                request.arguments.get("path", "."),
                allow_root=True,
            )
            if _is_sensitive_path(root, directory):
                raise WorkspacePathError("That workspace path is not available to the file tools.")
            if not directory.is_dir():
                raise WorkspacePathError("The requested workspace path is not a directory.")

            recursive = request.arguments.get("recursive", False)
            if not isinstance(recursive, bool):
                raise WorkspacePathError("'recursive' must be a boolean.")
            limit = request.arguments.get("limit", DEFAULT_LIST_LIMIT)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
                raise WorkspacePathError(f"'limit' must be an integer from 1 to {MAX_LIST_LIMIT}.")

            candidates = directory.rglob("*") if recursive else directory.iterdir()
            items = []
            for candidate in candidates:
                try:
                    resolved = candidate.resolve(strict=False)
                    resolved.relative_to(root)
                    if (
                        _is_sensitive_path(root, resolved)
                        or _is_sensitive_parts(candidate.relative_to(root).parts)
                    ):
                        continue
                    kind = "directory" if candidate.is_dir() else "file"
                    if kind == "file" and not candidate.is_file():
                        continue
                    items.append(
                        {
                            "path": candidate.relative_to(root).as_posix(),
                            "kind": kind,
                        }
                    )
                except (OSError, RuntimeError, ValueError):
                    # Skip unreadable entries and links that leave the workspace.
                    continue
                if len(items) >= limit:
                    break

            items.sort(key=lambda item: (item["kind"] != "directory", item["path"].lower()))
            return _tool_result(
                request,
                success=True,
                output={
                    "path": directory.relative_to(root).as_posix() or ".",
                    "items": items,
                    "count": len(items),
                    "truncated": len(items) >= limit,
                },
                start=start,
            )
        except (OSError, WorkspacePathError) as exc:
            return _tool_result(request, success=False, error=str(exc), start=start)


class ReadWorkspaceFileTool(_WorkspaceFileTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="filesystem.read_file",
            description=(
                "Read a UTF-8 text file inside the active project workspace. "
                "Credential files and files outside the workspace are blocked."
            ),
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}},
                "required": ["path"],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["read_file"],
                "category": "filesystem",
                "workspace_scoped": True,
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        import time

        start = time.perf_counter()
        try:
            root = self._root()
            requested_path = request.arguments.get("path")
            path = resolve_workspace_path(root, requested_path)
            if (
                _is_sensitive_path(root, path)
                or _is_sensitive_parts(
                    str(requested_path).replace("\\", "/").split("/")
                )
            ):
                raise WorkspacePathError("That file is not available to the file tools.")
            if not path.is_file():
                raise WorkspacePathError("The requested workspace file does not exist.")
            if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                raise WorkspacePathError(
                    f"Files larger than {MAX_TEXT_FILE_BYTES} bytes cannot be read by this tool."
                )
            with path.open("rb") as handle:
                raw = handle.read(MAX_TEXT_FILE_BYTES + 1)
            if len(raw) > MAX_TEXT_FILE_BYTES:
                raise WorkspacePathError(
                    f"Files larger than {MAX_TEXT_FILE_BYTES} bytes cannot be read by this tool."
                )
            content = raw.decode("utf-8")
            self._record_recent_file(root, path)
            return _tool_result(
                request,
                success=True,
                output={
                    "path": path.relative_to(root).as_posix(),
                    "content": content,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes_read": len(raw),
                },
                start=start,
            )
        except (OSError, UnicodeDecodeError, WorkspacePathError) as exc:
            return _tool_result(request, success=False, error=str(exc), start=start)


class WriteWorkspaceFileTool(_WorkspaceFileTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="filesystem.write_file",
            description=(
                "Create or update a UTF-8 text file inside the active project workspace. "
                "New files are create-only; updating an existing file requires the SHA-256 "
                "returned by filesystem.read_file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            risk_level="medium",
            metadata={
                "actions": ["write_file"],
                "category": "filesystem",
                "workspace_scoped": True,
                "atomic_write": True,
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        import time

        start = time.perf_counter()
        temp_path: Path | None = None
        try:
            root = self._root()
            requested_path = request.arguments.get("path")
            path = resolve_workspace_path(root, requested_path)
            if (
                _is_sensitive_path(root, path)
                or _is_sensitive_parts(
                    str(requested_path).replace("\\", "/").split("/")
                )
            ):
                raise WorkspacePathError("That file is not available to the file tools.")

            content = request.arguments.get("content")
            if not isinstance(content, str):
                raise WorkspacePathError("'content' must be a string.")
            payload = content.encode("utf-8")
            if len(payload) > MAX_TEXT_FILE_BYTES:
                raise WorkspacePathError(
                    f"Files larger than {MAX_TEXT_FILE_BYTES} bytes cannot be written by this tool."
                )

            expected = request.arguments.get("expected_sha256")
            if expected is not None and (
                not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)
            ):
                raise WorkspacePathError(
                    "'expected_sha256' must be a 64-character SHA-256 digest."
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            # Re-resolve after creating parents to catch links introduced in
            # any already-existing part of the path.
            path = resolve_workspace_path(root, request.arguments.get("path"))
            path.parent.resolve(strict=True).relative_to(root)

            exists = path.exists()
            previous_text = None
            previous_hash = None
            if exists:
                if not path.is_file():
                    raise WorkspacePathError("The requested workspace path is not a regular file.")
                if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                    raise WorkspacePathError(
                        f"Files larger than {MAX_TEXT_FILE_BYTES} bytes cannot be updated by this tool."
                    )
                if expected is None:
                    raise WorkspacePathError(
                        "The file already exists. Read it first and supply its expected_sha256 to update it."
                    )
                with path.open("rb") as handle:
                    existing = handle.read(MAX_TEXT_FILE_BYTES + 1)
                if len(existing) > MAX_TEXT_FILE_BYTES:
                    raise WorkspacePathError(
                        f"Files larger than {MAX_TEXT_FILE_BYTES} bytes cannot be updated by this tool."
                    )
                existing_hash = hashlib.sha256(existing).hexdigest()
                previous_hash = existing_hash
                previous_text = existing.decode("utf-8", errors="replace")
                if not hmac.compare_digest(existing_hash, expected.lower()):
                    raise WorkspacePathError(
                        "The file changed since it was read; read it again before updating."
                    )
            elif expected is not None:
                raise WorkspacePathError(
                    "The file no longer exists; omit expected_sha256 to create it."
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".asta-write-",
                dir=str(path.parent),
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, path)
            temp_path = None
            digest = hashlib.sha256(payload).hexdigest()
            self._record_recent_file(root, path)
            relative = path.relative_to(root).as_posix()
            change = text_change_summary(previous_text, content, relative)
            return _tool_result(
                request,
                success=True,
                output={
                    "path": relative,
                    "created": not exists,
                    "bytes_written": len(payload),
                    "sha256": digest,
                    **({"previous_sha256": previous_hash} if previous_hash else {}),
                    **change,
                },
                start=start,
            )
        except (OSError, RuntimeError, WorkspacePathError) as exc:
            return _tool_result(request, success=False, error=str(exc), start=start)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass