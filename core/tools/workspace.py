"""Workspace-scoped text file capabilities.

All paths are relative to the active workspace. These tools intentionally
provide a small, auditable surface for project work: list, read, and
create/update text files. They do not delete or move files.
"""

from __future__ import annotations

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
            return _tool_result(
                request,
                success=True,
                output={
                    "path": path.relative_to(root).as_posix(),
                    "created": not exists,
                    "bytes_written": len(payload),
                    "sha256": digest,
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