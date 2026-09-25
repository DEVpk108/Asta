DIAGNOSIS_CATEGORIES = {
    "authentication": "The action cannot proceed until the user signs in or completes OAuth.",
    "setup_required": "The integration or capability is not configured yet and requires setup before execution can continue.",
    "authorization": "The action reached an access or permission boundary that the current credentials cannot cross.",
    "transient": "The failure looks temporary, such as timeout, network interruption, service busy, or connection reset.",
    "missing_capability": "A required tool, integration, API capability, or registered capability is unavailable.",
    "invalid_target": "The requested application, file, track, URL, or other target cannot be found or is invalid.",
    "state_mismatch": "The target exists but the current environment or application state does not match the action precondition.",
    "unknown": "The available evidence is insufficient to identify a more specific failure category.",
}

RECOVERY_ACTIONS = {
    "retry": "Retry the same step when the failure is plausibly transient.",
    "wait_for_user": "Pause because user authentication, authorization, or explicit confirmation is required.",
    "replan": "Change the plan or choose a different execution path.",
    "fail": "Stop because continuing is not justified by the available evidence.",
}


def build_diagnosis_questions() -> dict:
    return {
        "category": {
            "type": "choice",
            "instructions": (
                "Classify the execution failure into exactly one bounded category. "
                "Use the observed error, tool, output, goal, and recovery context. "
                "Do not invent a category."
            ),
            "criteria": DIAGNOSIS_CATEGORIES,
        },
        "recommended_action": {
            "type": "choice",
            "instructions": (
                "Which bounded recovery direction best matches the diagnosed failure? "
                "This is a recommendation only; the runtime keeps its own safety policy."
            ),
            "criteria": RECOVERY_ACTIONS,
        },
        "requires_user": {
            "type": "noul",
            "instructions": (
                "Does progress require a user-only boundary such as login, MFA, "
                "CAPTCHA, account authorization, or explicit confirmation?"
            ),
            "criteria": {
                "true": "A user-only security or confirmation step is required.",
                "false": "The task can proceed without a user-only security boundary.",
            },
        },
    }
