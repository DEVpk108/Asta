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

    def __init__(self, media_manager=None, *, poll_attempts: int = 3, poll_delay: float = 0.4):
        self.media_manager = media_manager
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

        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            summary=f"No verifier is registered for '{verification}'.",
            source="guard",
            task_id=task_id,
            step_id=step_id,
            metadata={"verification": verification},
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
