"""Clean up spoken-command transcripts before routing.

Two problems show up with streaming multilingual STT on Indian-accented
English:

* The wake phrase bleeds into the command ("Upyasta open Chrome",
  "ओप्यास्टा ओपन क्रोम"), so the router sees an unknown first word.
* English speech is written in Devanagari ("ओपन क्रोम" for "open Chrome"),
  and short English commands get clipped while the Hindi decode keeps them.

This module is deliberately dictionary based and dependency free: it maps a
bounded vocabulary of command words and common application names, strips
wake-word remnants, and rewrites simple Hindi/Hinglish command grammar
("क्रोम खोलो", "chrome band karo") into the English form the deterministic
router already understands.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Devanagari spellings of English words that STT produces for English speech.
_DEVANAGARI_ENGLISH = {
    "open": "ओपन ओपेन ओप्पन ओपन्ड",
    "close": "क्लोज क्लोज़ क्लोस क्लोज्ड",
    "launch": "लॉन्च लांच लाँच",
    "start": "स्टार्ट स्टाट",
    "stop": "स्टॉप स्टाप स्टोप",
    "run": "रन",
    "play": "प्ले प्लेय",
    "pause": "पॉज पॉज़ पौज पोज़",
    "resume": "रिज्यूम रिज़्यूम",
    "next": "नेक्स्ट",
    "previous": "प्रीवियस प्रिवियस",
    "skip": "स्किप",
    "search": "सर्च",
    "for": "फॉर फ़ॉर फोर",
    "on": "ऑन",
    "in": "इन",
    "the": "द दि",
    "a": "अ",
    "and": "एंड ऐंड एन्ड",
    "then": "देन दैन",
    "please": "प्लीज प्लीज़",
    "take": "टेक",
    "screenshot": "स्क्रीनशॉट स्क्रीनशोट स्क्रीनशाट",
    "screen": "स्क्रीन",
    "mute": "म्यूट",
    "unmute": "अनम्यूट",
    "volume": "वॉल्यूम वोल्यूम वॉल्युम",
    "up": "अप",
    "down": "डाउन",
    "song": "सॉन्ग सोंग",
    "music": "म्यूजिक म्यूज़िक",
    "video": "वीडियो विडियो",
    "new": "न्यू",
    "tab": "टैब",
    "window": "विंडो",
    "app": "ऐप एप",
    "application": "एप्लीकेशन एप्लिकेशन ऐप्लिकेशन",
    "scroll": "स्क्रॉल स्क्रोल",
    "click": "क्लिक",
    "type": "टाइप",
    "note": "नोट",
    "notes": "नोट्स",
    "okay": "ओके",
    "hey": "हे",
    "hello": "हेलो हैलो",
    "asta": "अस्टा आस्टा एस्टा अस्ता आस्ता",
    # Applications and providers.
    "chrome": "क्रोम क्रो क्रोम्ब क्रोमे क्रम क्रोन",
    "camera": "कैमरा कैमेरा केमरा",
    "spotify": "स्पॉटिफाई स्पोटिफाई स्पॉटिफ़ाई स्पोटीफाई स्पॉटीफाई",
    "youtube": "यूट्यूब यूटयूब युट्यूब यूट्युब",
    "google": "गूगल",
    "notepad": "नोटपैड नोटपेड",
    "calculator": "कैलकुलेटर कैलक्यूलेटर केलकुलेटर",
    "settings": "सेटिंग्स सेटिंग",
    "file": "फाइल फ़ाइल",
    "explorer": "एक्सप्लोरर",
    "browser": "ब्राउजर ब्राउज़र",
    "edge": "एज",
    "firefox": "फायरफॉक्स फ़ायरफ़ॉक्स",
    "whatsapp": "व्हाट्सएप व्हाट्सऐप वॉट्सऐप व्हाट्सअप वाट्सएप",
    "telegram": "टेलीग्राम टेलिग्राम",
    "discord": "डिस्कॉर्ड डिस्कोर्ड",
    "word": "वर्ड",
    "excel": "एक्सेल",
    "powerpoint": "पावरपॉइंट पावरपॉइन्ट",
    "vs": "वीएस",
    "code": "कोड",
    "terminal": "टर्मिनल",
    "paint": "पेंट",
    "outlook": "आउटलुक",
    "teams": "टीम्स",
    "steam": "स्टीम",
    "vlc": "वीएलसी",
    "photos": "फोटोज फ़ोटोज़ फोटोस",
    "gmail": "जीमेल",
}

DEVANAGARI_TO_ENGLISH = {
    spelling: english
    for english, spellings in _DEVANAGARI_ENGLISH.items()
    for spelling in spellings.split()
}

# Hindi command verbs (Devanagari and romanized Hinglish) -> English action.
_HINDI_ACTIONS = (
    ("open", r"(?:खोलो|खोल\s+दो|खोल\s+दीजिए|खोलिए|खोलिये|खोलें|खोलना|ओपन\s+करो|ओपन\s+कर\s+दो"
             r"|kholo|khol\s+do|kholiye|kholna|open\s+karo|open\s+kar\s+do)"),
    ("close", r"(?:बंद\s+करो|बंद\s+कर\s+दो|बंद\s+कीजिए|बंद\s+करें|बन्द\s+करो|क्लोज\s+करो"
              r"|band\s+karo|band\s+kar\s+do|bandh\s+karo|close\s+karo|close\s+kar\s+do)"),
    ("play", r"(?:चलाओ|चला\s+दो|बजाओ|बजा\s+दो|लगाओ|लगा\s+दो|प्ले\s+करो"
             r"|chalao|chala\s+do|bajao|baja\s+do|lagao|laga\s+do|play\s+karo)"),
    ("search", r"(?:सर्च\s+करो|ढूंढो|ढूँढो|खोजो|search\s+karo|dhundo|khojo)"),
)
_HINDI_ACTION_PATTERNS = tuple(
    (action, re.compile(rf"^(?P<target>.+?)\s+{pattern}(?:\s+(?:ना|न|na|please|प्लीज))?$", re.IGNORECASE))
    for action, pattern in _HINDI_ACTIONS
)
# Hindi joiners between clauses ("chrome kholo aur camera kholo").
_HINDI_AND = re.compile(r"\s+(?:और|फिर|उसके\s+बाद|aur|phir|fir)\s+", re.IGNORECASE)
_HINDI_OBJECT_MARKERS = re.compile(r"\s+(?:को|ko)$", re.IGNORECASE)
# Hindi nouns that commonly appear as command targets.
_HINDI_TARGETS = {
    "गाना": "music", "गाने": "music", "संगीत": "music",
    "gaana": "music", "gana": "music", "gaane": "music",
}

# Wake-word remnants at the start of a transcript.
_LATIN_WAKE_TOKEN = re.compile(
    r"^(?:hey|hi|hay|hai|he|hello|ok|okay|up|ug|uh|ap|op|ab|a)?y?aa?st(?:a|aa|ah|ha|haa)$"
)
_LATIN_WAKE_LEADS = {"hey", "hi", "hay", "hai", "he", "hello", "ok", "okay", "a", "wake"}
_LATIN_WAKE_NAMES = re.compile(r"^(?:aa?st(?:a|aa|ah|ha|er)|esta|ista|astra|asthaa?)$")
_DEVANAGARI_WAKE_TOKEN = re.compile(r"(?:्यास्ट|यास्ट)[ा]?$")
_DEVANAGARI_WAKE_NAMES = {"अस्टा", "आस्टा", "एस्टा", "अस्ता", "आस्ता", "अस्टर", "आस्टर"}
_DEVANAGARI_WAKE_LEADS = {"हे", "हेलो", "हैलो", "ओके", "हाय"}

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Rough Devanagari romanisation, used only to fuzzy-match an unknown
# Devanagari word against the transliteration vocabulary above.
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ड़": "r", "ढ़": "rh",
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "फ़": "f",
}
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऋ": "ri",
}
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॉ": "o", "ृ": "ri",
}
_VIRAMA = "\u094d"
_NASALS = {"ं": "n", "ँ": "n", "ः": "h"}


def romanize(token: str) -> str:
    """Very small Hindi romaniser (schwa dropped at the end of the word)."""
    token = unicodedata.normalize("NFC", token)
    out = []
    pending_schwa = False
    for char in token:
        if char in _CONSONANTS:
            if pending_schwa:
                out.append("a")
            out.append(_CONSONANTS[char])
            pending_schwa = True
        elif char in _MATRAS:
            out.append(_MATRAS[char])
            pending_schwa = False
        elif char == _VIRAMA:
            pending_schwa = False
        elif char in _VOWELS:
            if pending_schwa:
                out.append("a")
            out.append(_VOWELS[char])
            pending_schwa = False
        elif char in _NASALS:
            if pending_schwa:
                out.append("a")
            out.append(_NASALS[char])
            pending_schwa = False
        elif char == "\u093c":  # nukta
            continue
        else:
            if pending_schwa:
                out.append("a")
            pending_schwa = False
            out.append(char)
    return "".join(out)


_ROMAN_VOCABULARY = tuple(
    (romanize(spelling), english)
    for spelling, english in DEVANAGARI_TO_ENGLISH.items()
    if len(spelling) >= 3
)


def fuzzy_devanagari_english(token: str, *, minimum: float = 0.75) -> str | None:
    """Closest vocabulary word for an unknown Devanagari token, if close."""
    if len(token) < 3 or not _DEVANAGARI.search(token):
        return None
    roman = romanize(token)
    best, best_score = None, 0.0
    for candidate, english in _ROMAN_VOCABULARY:
        score = SequenceMatcher(None, roman, candidate, autojunk=False).ratio()
        if score > best_score:
            best, best_score = english, score
    return best if best_score >= minimum else None


def words(text: str) -> list[str]:
    """Split text into lowercase words, keeping Devanagari vowel signs."""
    value = "".join(
        ch if unicodedata.category(ch)[0] in "LMN" else " "
        for ch in str(text or "").lower()
    )
    return value.split()


def _bare(token: str) -> str:
    return "".join(
        ch for ch in token.lower() if unicodedata.category(ch)[0] in "LMN"
    )


def strip_wake_remnant(text: str) -> str:
    """Remove a leading wake phrase the STT merged into the command.

    Only strips when something follows, so a bare "Asta" stays intact.
    """
    value = " ".join(str(text or "").split())
    tokens = value.split(" ")
    if len(tokens) < 2:
        return value

    first = _bare(tokens[0])
    second = _bare(tokens[1]) if len(tokens) > 1 else ""

    drop = 0
    if first == "wake" and second == "up" and len(tokens) > 3 and (
        _LATIN_WAKE_NAMES.match(_bare(tokens[2]))
        or _LATIN_WAKE_TOKEN.match(_bare(tokens[2]))
    ):
        drop = 3
    elif first in _LATIN_WAKE_LEADS and len(tokens) > 2 and (
        _LATIN_WAKE_NAMES.match(second) or _LATIN_WAKE_TOKEN.match(second)
    ):
        drop = 2
    elif _LATIN_WAKE_TOKEN.match(first) or _LATIN_WAKE_NAMES.match(first):
        drop = 1
    elif first in _DEVANAGARI_WAKE_LEADS and len(tokens) > 2 and (
        second in _DEVANAGARI_WAKE_NAMES or _DEVANAGARI_WAKE_TOKEN.search(second)
    ):
        drop = 2
    elif first in _DEVANAGARI_WAKE_NAMES or _DEVANAGARI_WAKE_TOKEN.search(first):
        drop = 1

    if not drop:
        return value
    remainder = " ".join(tokens[drop:]).lstrip(" ,.!?;:-")
    return remainder or value


def devanagari_english(
    text: str,
    *,
    fuzzy: bool = False,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> tuple[str, float]:
    """Map a Devanagari transcript of English speech back to English.

    Returns ``(english_text, coverage)`` where coverage is the share of words
    found in the transliteration vocabulary. Unknown words are kept as-is.
    """
    tokens = words(text)
    if not tokens:
        return "", 0.0
    mapped = []
    known = 0
    for token in tokens:
        english = DEVANAGARI_TO_ENGLISH.get(token)
        if english is None and fuzzy and token not in exclude:
            english = fuzzy_devanagari_english(token)
        if english is None and not _DEVANAGARI.search(token):
            # Latin words inside a Devanagari transcript are already English.
            english = token
        if english is not None:
            known += 1
            mapped.append(english)
        else:
            mapped.append(token)
    return " ".join(mapped), known / len(tokens)


def canonicalize_command(text: str) -> str:
    """Rewrite Hindi/Hinglish or Devanagari-English commands into English.

    Returns the input unchanged when nothing applies, so ordinary sentences
    pass straight through to the router.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return value

    has_devanagari = bool(_DEVANAGARI.search(value))
    working = value
    if has_devanagari:
        # Replace known transliterated English words in place.
        parts = []
        for token in working.split(" "):
            bare = _bare(token)
            parts.append(DEVANAGARI_TO_ENGLISH.get(bare, token))
        working = " ".join(parts)

    clauses = _HINDI_AND.split(working)
    rewritten = []
    changed = False
    for clause in clauses:
        clause = clause.strip(" ,.!?;:।")
        converted = None
        for action, pattern in _HINDI_ACTION_PATTERNS:
            match = pattern.match(clause)
            if match:
                target = _HINDI_OBJECT_MARKERS.sub("", match.group("target")).strip()
                target = _HINDI_TARGETS.get(target.lower(), target)
                if target:
                    converted = f"{action} {target}"
                break
        if converted is not None:
            changed = True
            rewritten.append(converted)
        else:
            rewritten.append(clause)

    if changed:
        return " then ".join(rewritten)
    if has_devanagari and working != value:
        # Fully transliterated English such as "ओपन क्रोम" -> "open chrome".
        _, coverage = devanagari_english(value)
        if coverage >= 0.6:
            return working
    return value
