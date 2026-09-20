from __future__ import annotations

from ..contracts import SkillDescriptor


SOFTWARE_DEBUGGING_SKILL = SkillDescriptor(
    name="software_debugging",
    description="A disciplined workflow for diagnosing and fixing software problems.",
    instructions=(
        "1. Reproduce the problem and identify the smallest failing case. "
        "2. Inspect the relevant code, logs, inputs, and recent changes. "
        "3. Form a concrete hypothesis before editing code. "
        "4. Make the smallest reversible change that tests the hypothesis. "
        "5. Run the narrowest useful verification first, then broader tests. "
        "6. Record the observed result and remaining uncertainty."
    ),
    triggers=("debug", "fix", "bug", "error", "exception", "test", "failing"),
    priority=20,
    metadata={
        "domains": ("software", "coding", "python"),
        "actions": ("debug", "fix", "test"),
    },
)


ELECTRONICS_DEBUGGING_SKILL = SkillDescriptor(
    name="electronics_debugging",
    description="A structured workflow for diagnosing electronics and hardware faults.",
    instructions=(
        "1. Confirm the expected behavior and power requirements. "
        "2. Verify supply voltage, ground continuity, and basic wiring. "
        "3. Trace the signal path from source to destination. "
        "4. Measure at subsystem boundaries and isolate the smallest failing section. "
        "5. Change one variable at a time and record measurements. "
        "6. Re-test after each change and keep the working configuration reproducible."
    ),
    triggers=("electronics", "hardware", "circuit", "sensor", "wiring", "debug"),
    priority=15,
    metadata={
        "domains": ("electronics", "hardware"),
        "actions": ("debug",),
    },
)


def register_builtin_skills(manager) -> None:
    """Register A.S.T.A.'s initial reusable problem-solving skills."""
    for skill in (
        SOFTWARE_DEBUGGING_SKILL,
        ELECTRONICS_DEBUGGING_SKILL,
    ):
        try:
            manager.register(skill)
        except ValueError:
            # Startup should remain idempotent if a caller registered a
            # built-in skill explicitly before invoking this helper.
            pass
