"""Canonical semantic-inspector outcome, shared by cognitive and fast paths."""
from typing import Any


def is_verified_inspection_output(output: Any) -> bool:
    """Respect the inspector's final verdict, not raw model/visibility flags.

    VisionInspectTool applies confidence and title guards before setting the
    top-level ``verified`` boolean. Nested analysis must not override it.
    """
    return (
        isinstance(output, dict)
        and output.get("verified") is True
        and ("visual_match" not in output or output["visual_match"] is True)
    )