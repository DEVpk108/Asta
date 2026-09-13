"""
A.S.T.A. A.I Package
"""

from .ai_module import AIModule


CREATOR_NAME = "Mr. PRASANT KUMAR"
_CREATOR_QUESTION_VARIANTS = {
    "who created you",
    "who made you",
    "who built you",
    "who developed you",
    "who is your creator",
    "who created asta",
    "who made asta",
    "who built asta",
    "who developed asta",
}

_original_on_user_message = AIModule.on_user_message


def _on_user_message_with_creator_identity(self, text):
    normalized = " ".join(str(text).strip().lower().split()).rstrip(" .!?;:")
    if normalized in _CREATOR_QUESTION_VARIANTS:
        print("[AI] Creator identity response", flush=True)
        self._emit_assistant_text(f"I was created by {CREATOR_NAME}.")
        return

    _original_on_user_message(self, text)


AIModule.on_user_message = _on_user_message_with_creator_identity


__all__ = [
    "AIModule",
    "CREATOR_NAME",
]
