"""Conservative consent language shared by voice and typed approval paths.

This module never grants authority: the handler still binds a decision to a
pending request and the tool runtime enforces expiry and one-time consumption.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_confirmation_transcript(text) -> str:
    """Keep the complete reply; never turn a first word into a decision."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.translate(str.maketrans({"’": "'", "‘": "'"}))
    # Preserve question marks: "yes?" is not unambiguous consent.
    return " ".join(value.casefold().split()).rstrip(" .!;:,")


_YES = frozenset({
    "yes", "yeah", "yep", "yup", "confirm", "confirmed", "approve",
    "approved", "proceed", "haan", "han", "ha", "हाँ", "हां", "हा",
})
_NO = frozenset({
    "no", "nope", "nah", "not", "dont", "don't", "cannot", "can't",
    "cant", "cancel", "stop", "reject", "decline", "never", "wait",
    "nothing", "later", "nahi", "nahin", "mat", "नहीं", "नही", "मत",
})
_EXPLICIT = frozenset({
    "go ahead", "do it", "i confirm", "i approve", "i just said yes",
    "okay go ahead", "okay do it", "sure go ahead", "sure do it",
})
_POLITE_APPROVAL = re.compile(
    r"(?:yes|yeah|yep|yup|confirm|confirmed|approve|approved|proceed|"
    r"haan|han|ha|हाँ|हां|हा)"
    r"(?: please)?(?: (?:go ahead|do it|proceed|continue))?(?: please)?"
)


def classify_approval(text) -> tuple[bool, bool]:
    """Return (approve, reject), failing closed on mixed/uncertain language.

    A whole-utterance grammar is intentional. Finding "yes" inside a question,
    conditional, reported statement or new command must not authorize a tool.
    Mixed replies remain undecided; the ordinary handler cancels the pending
    request, while background approvals retain their explicit-command policy.
    """
    value = normalize_confirmation_transcript(text)
    words = re.findall(r"[\w'\u0900-\u097F]+", value)
    if not words or len(words) > 10:
        return False, False
    has_yes = any(word in _YES for word in words)
    has_no = any(word in _NO for word in words)
    if has_no:
        return False, not has_yes
    if "?" in value:
        return False, False
    if value in _YES or value in _EXPLICIT or _POLITE_APPROVAL.fullmatch(value):
        return True, False
    return False, False