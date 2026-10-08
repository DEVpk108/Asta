from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class VerificationResult:
    """Evidence-backed outcome for one planned step or task goal."""

    status: VerificationStatus
    summary: str
    source: str
    task_id: str | None = None
    step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "source": self.source,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "metadata": dict(self.metadata),
        }


class VerificationEngine:
    """Run bounded, capability-specific post-action verification.

    Tool success is still enough for capabilities that do not declare an
    external verifier. Capabilities that do declare one must produce observed
    state before TaskRuntime can report the task as complete.
    """

    def __init__(
        self,
        media_manager=None,
        application_manager=None,
        *,
        poll_attempts: int = 3,
        poll_delay: float = 0.2,
    ):
        self.media_manager = media_manager
        self.application_manager = application_manager
        self.poll_attempts = max(1, int(poll_attempts))
        self.poll_delay = max(0.0, float(poll_delay))

    def verify(self, task, step, result) -> VerificationResult:
        task_id = getattr(task, "id", None)
        step_id = getattr(step, "id", None) if step is not None else None
        verification = str(
            (getattr(step, "metadata", {}) or {}).get("verification") or ""
        ).strip().lower()

        if not verification:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                summary="Tool execution completed successfully.",
                source="tool_result",
                task_id=task_id,
                step_id=step_id,
            )

        if verification == "media.playback":
            return self._verify_media_playback(
                task,
                step,
                result,
            )

        if verification == "application.running":
            return self._verify_application_running(
                task,
                step,
                result,
            )

        if verification == "command.output":
            return self._verify_command_output(
                task,
                step,
                result,
            )

        if verification == "filesystem.file_content":
            return self._verify_file_content(
                task,
                step,
                result,
            )

        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            summary=f"No verifier is registered for '{verification}'.",
            source="guard",
            task_id=task_id,
            step_id=step_id,
            metadata={"verification": verification},
        )

    def _verify_application_running(self, task, step, result) -> VerificationResult:
        """Verify an application launch with OS process state, not vision."""
        task_id = getattr(task, "id", None)
        step_id = getattr(step, "id", None)
        manager = self.application_manager
        checker = getattr(manager, "is_application_running", None)
        if not callable(checker):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Application runtime verification is unavailable.",
                source="application_manager",
                task_id=task_id,
                step_id=step_id,
            )

        metadata = getattr(step, "metadata", {}) or {}
        output = getattr(result, "output", None)
        output = output if isinstance(output, dict) else {}
        query = str(
            metadata.get("target")
            or output.get("target")
            or output.get("resolved_target")
            or ""
        ).strip()

        if not query:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Application verification has no target.",
                source="application_manager",
                task_id=task_id,
                step_id=step_id,
            )

        last_error = None
        for attempt in range(1, self.poll_attempts + 1):
            try:
                if bool(checker(query)):
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        summary=f"Confirmed that {query} is running.",
                        source="application_manager",
                        task_id=task_id,
                        step_id=step_id,
                        metadata={
                            "query": query,
                            "attempt": attempt,
                        },
                    )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.poll_attempts:
                time.sleep(self.poll_delay)

        summary = (
            f"Could not confirm that {query} is running."
            if not last_error
            else f"Application runtime check failed: {last_error}"
        )
        return VerificationResult(
            status=VerificationStatus.FAILED,
            summary=summary,
            source="application_manager",
            task_id=task_id,
            step_id=step_id,
            metadata={"query": query},
        )

    @staticmethod
    def _verify_command_output(task, step, result) -> VerificationResult:
        """Compare an observed command result with explicit plan conditions."""
        task_id = getattr(task, "id", None)
        step_id = getattr(step, "id", None)
        metadata = getattr(step, "metadata", {}) or {}
        output = getattr(result, "output", None)
        output = output if isinstance(output, dict) else {}

        expected_stdout = metadata.get("expected_stdout")
        if not isinstance(expected_stdout, str):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Command-output verification has no expected_stdout condition.",
                source="command_result",
                task_id=task_id,
                step_id=step_id,
            )

        try:
            raw_expected_returncode = metadata.get("expected_returncode", 0)
            if isinstance(raw_expected_returncode, bool):
                raise ValueError("boolean return codes are not valid")
            expected_returncode = int(raw_expected_returncode)
            actual_returncode = int(output["returncode"])
        except (KeyError, TypeError, ValueError):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Command-output verification lacks a valid process return code.",
                source="command_result",
                task_id=task_id,
                step_id=step_id,
            )

        actual_stdout = output.get("stdout")
        if not isinstance(actual_stdout, str):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Command-output verification lacks captured stdout.",
                source="command_result",
                task_id=task_id,
                step_id=step_id,
            )

        match_mode = str(metadata.get("stdout_match") or "exact").strip().lower()
        expected = expected_stdout.rstrip("\r\n")
        actual = actual_stdout.rstrip("\r\n")
        if match_mode == "exact":
            output_matches = actual == expected
        elif match_mode == "contains":
            output_matches = expected in actual
        else:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Command-output verification supports only exact or contains matching.",
                source="command_result",
                task_id=task_id,
                step_id=step_id,
                metadata={"stdout_match": match_mode},
            )

        verified = actual_returncode == expected_returncode and output_matches
        summary = (
            "Observed the expected command exit code and stdout."
            if verified
            else "The command result did not match the expected exit code or stdout."
        )
        return VerificationResult(
            status=(
                VerificationStatus.VERIFIED
                if verified
                else VerificationStatus.FAILED
            ),
            summary=summary,
            source="command_result",
            task_id=task_id,
            step_id=step_id,
            metadata={
                "expected_returncode": expected_returncode,
                "actual_returncode": actual_returncode,
                "expected_stdout": expected,
                "actual_stdout": actual,
                "stdout_match": match_mode,
            },
        )

    @staticmethod
    def _verify_file_content(task, step, result) -> VerificationResult:
        """Verify a workspace readback against the plan's exact file condition."""
        task_id = getattr(task, "id", None)
        step_id = getattr(step, "id", None)
        metadata = getattr(step, "metadata", {}) or {}
        output = getattr(result, "output", None)
        output = output if isinstance(output, dict) else {}

        expected_path = metadata.get("expected_path")
        expected_content = metadata.get("expected_content")
        actual_path = output.get("path")
        actual_content = output.get("content")
        if not isinstance(expected_path, str) or not isinstance(expected_content, str):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="File-content verification lacks an expected path or content.",
                source="filesystem.read_file",
                task_id=task_id,
                step_id=step_id,
            )
        if not isinstance(actual_path, str) or not isinstance(actual_content, str):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="File-content verification lacks readable file evidence.",
                source="filesystem.read_file",
                task_id=task_id,
                step_id=step_id,
            )

        normalize_path = lambda value: value.replace("\\", "/").strip("/")
        path_matches = normalize_path(actual_path) == normalize_path(expected_path)
        content_matches = actual_content == expected_content
        verified = path_matches and content_matches
        return VerificationResult(
            status=(
                VerificationStatus.VERIFIED
                if verified
                else VerificationStatus.FAILED
            ),
            summary=(
                "The read-back file path and contents match the planned condition."
                if verified
                else "The read-back file path or contents did not match the planned condition."
            ),
            source="filesystem.read_file",
            task_id=task_id,
            step_id=step_id,
            metadata={
                "expected_path": expected_path,
                "actual_path": actual_path,
                "path_matches": path_matches,
                "content_matches": content_matches,
                "sha256": output.get("sha256"),
            },
        )

    def _verify_media_playback(self, task, step, result) -> VerificationResult:
        task_id = getattr(task, "id", None)
        step_id = getattr(step, "id", None)
        metadata = getattr(step, "metadata", {}) or {}
        provider = str(metadata.get("provider") or "").strip().lower()
        query = str(metadata.get("query") or "").strip()
        output = getattr(result, "output", None)
        output = output if isinstance(output, dict) else {}
        expected_uri = str(output.get("uri") or "").strip() or None

        if not provider or not query:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="Media playback verification lacks provider or query context.",
                source="guard",
                task_id=task_id,
                step_id=step_id,
            )

        observer = getattr(self.media_manager, "verify_playback", None)
        if not callable(observer):
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                summary="The media manager has no playback observer.",
                source="guard",
                task_id=task_id,
                step_id=step_id,
                metadata={"provider": provider, "query": query},
            )

        last_observation: dict[str, Any] | None = None
        for attempt in range(1, self.poll_attempts + 1):
            try:
                observation = observer(
                    provider,
                    query,
                    expected_uri=expected_uri,
                )
            except Exception as exc:
                observation = {
                    "status": VerificationStatus.UNKNOWN.value,
                    "summary": f"Playback observation failed: {type(exc).__name__}: {exc}",
                }

            if not isinstance(observation, dict):
                observation = {
                    "status": VerificationStatus.UNKNOWN.value,
                    "summary": "Playback observer returned invalid data.",
                }

            last_observation = dict(observation)
            status = str(
                observation.get("status")
                or VerificationStatus.UNKNOWN.value
            ).strip().lower()

            if status == VerificationStatus.VERIFIED.value:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    summary=str(
                        observation.get("summary")
                        or "Observed the expected track playing."
                    ),
                    source="media_observer",
                    task_id=task_id,
                    step_id=step_id,
                    metadata={
                        "provider": provider,
                        "query": query,
                        "attempt": attempt,
                        "observation": observation,
                    },
                )

            if attempt < self.poll_attempts:
                time.sleep(self.poll_delay)

        final_status = str(
            (last_observation or {}).get("status")
            or VerificationStatus.UNKNOWN.value
        ).strip().lower()
        mapped_status = (
            VerificationStatus.FAILED
            if final_status == VerificationStatus.FAILED.value
            else VerificationStatus.UNKNOWN
        )

        return VerificationResult(
            status=mapped_status,
            summary=str(
                (last_observation or {}).get("summary")
                or "The expected playback state was not observed."
            ),
            source="media_observer",
            task_id=task_id,
            step_id=step_id,
            metadata={
                "provider": provider,
                "query": query,
                "expected_uri": expected_uri,
                "observation": last_observation or {},
            },
        )
