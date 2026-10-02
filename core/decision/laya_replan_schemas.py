REPLAN_STRATEGIES = {
    "wait_for_user": (
        "Pause because the failure requires user authentication, authorization, "
        "MFA, CAPTCHA, or explicit confirmation."
    ),
    "restore_state": (
        "Restore a likely missing application/environment precondition, then retry "
        "the failed step."
    ),
    "rebuild_plan": (
        "Rebuild the remaining plan from the original goal and current context."
    ),
    "fail": (
        "Do not continue because the available evidence does not support a safe repair."
    ),
}


def build_replan_questions() -> dict:
    return {
        "strategy": {
            "type": "choice",
            "instructions": (
                "Choose exactly one bounded repair strategy for the failed task step. "
                "Use the diagnosis and step metadata. Do not invent a new strategy."
            ),
            "criteria": REPLAN_STRATEGIES,
        }
    }
