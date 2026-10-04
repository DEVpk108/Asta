"""Instant answers that never need the LLM (System 1).

"What is two plus two" should not wait on a 4B model: simple arithmetic is
parsed and computed locally in microseconds.
"""

from __future__ import annotations

import math
import re

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "lakh": 100_000, "crore": 10_000_000}

_OPERATORS = (
    (r"to\s+the\s+power\s+of|power|\^|\*\*", "**"),
    (r"multiplied\s+by|times|into|x|\*|×", "*"),
    (r"divided\s+by|over|by|/|÷", "/"),
    (r"plus|and|\+", "+"),
    (r"minus|less|-|−", "-"),
    (r"mod|modulo|%", "%"),
)
_LEAD = re.compile(
    r"^(?:hey\s+)?(?:please\s+)?(?:(?:what|how\s+much)\s+(?:is|are|'s)|what's|calculate|compute|"
    r"tell\s+me|solve|can\s+you\s+(?:tell\s+me|calculate))?\s*",
    re.IGNORECASE,
)


def _number(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    if len(tokens) == 1:
        try:
            return float(tokens[0].replace(",", ""))
        except ValueError:
            pass
    total, current, seen = 0, 0, False
    point = None
    for token in tokens:
        if token == "point":
            point = []
            continue
        if point is not None:
            if token in _UNITS and _UNITS[token] < 10:
                point.append(str(_UNITS[token]))
                continue
            return None
        if token in _UNITS:
            current += _UNITS[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current = max(current, 1) * 100
        elif token in _SCALES:
            total += max(current, 1) * _SCALES[token]
            current = 0
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            current += float(token)
        else:
            return None
        seen = True
    if not seen:
        return None
    value = total + current
    if point:
        value = float(f"{int(value)}.{''.join(point)}")
    return float(value)


def _format(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}".replace(",", ",")
    return f"{value:.4f}".rstrip("0").rstrip(".")


def quick_math(text: str) -> str | None:
    """Answer simple spoken arithmetic, or None."""
    value = str(text or "").strip().lower().rstrip("?.! ")
    value = _LEAD.sub("", value).strip()
    root = re.fullmatch(r"(?:the\s+)?(square|cube)\s+root\s+of\s+(.+)", value)
    if root:
        radicand = _number(root.group(2).replace("-", " ").split())
        if radicand is None or (root.group(1) == "square" and radicand < 0):
            return None
        if root.group(1) == "square":
            result = math.sqrt(radicand)
        else:
            result = round(radicand ** (1 / 3), 9)
        return f"That's {_format(result)}."
    value = re.sub(r"(\d)\s*([+\-*/×÷^%x])\s*(\d)", r"\1 \2 \3", value)
    value = re.sub(r"\bsquared\b", "** 2", value)
    value = re.sub(r"\bcubed\b", "** 3", value)
    if not value or not re.search(r"\d|\b(?:" + "|".join(list(_UNITS) + list(_TENS)) + r")\b", value):
        return None
    tokens = value.replace("-", " - ").split()
    pieces: list[str] = []
    number: list[str] = []
    expression = []
    i = 0
    while i < len(tokens):
        matched = None
        for pattern, symbol in _OPERATORS:
            for width in (4, 3, 2, 1):
                chunk = " ".join(tokens[i:i + width])
                if re.fullmatch(pattern, chunk) and not (symbol == "+" and chunk == "and" and not number):
                    matched = (symbol, width)
                    break
            if matched:
                break
        if matched and number:
            # "one hundred and five" is a number, not an addition.
            if matched[0] == "+" and tokens[i] == "and" and number[-1] == "hundred":
                i += 1
                continue
            parsed = _number(number)
            if parsed is None:
                return None
            expression.append(repr(parsed))
            expression.append(matched[0])
            pieces.append(" ".join(number))
            number = []
            i += matched[1]
            continue
        number.append(tokens[i])
        i += 1
    parsed = _number(number)
    if parsed is None or len(expression) < 2:
        return None
    expression.append(repr(parsed))
    if len(expression) > 15:
        return None
    for index, token in enumerate(expression):
        if token == "**" and abs(float(expression[index + 1])) > 64:
            return None
    try:
        result = eval(" ".join(expression), {"__builtins__": {}}, {})  # noqa: S307 - digits and operators only
    except (ZeroDivisionError, OverflowError):
        return "That can't be calculated, because it divides by zero."
    except Exception:
        return None
    if not isinstance(result, (int, float)) or abs(result) > 1e15:
        return None
    return f"That's {_format(float(result))}."
