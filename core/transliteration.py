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

import json
import os
import re
import unicodedata
from pathlib import Path
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
    """Rewrite noisy spoken commands into the English form the router knows.

    Handles Hindi/Hinglish grammar, Devanagari-English, misheard app names
    and a missing "then" between two commands. Ordinary sentences pass
    through unchanged.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return value
    return split_implicit_commands(
        apply_target_aliases(strip_transitions(fix_verb_typos(_canonicalize_language(value))))
    )


def _canonicalize_language(value: str) -> str:

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


# --------------------------------------------------------------------------
# Command-target aliases for consistent mishearings
# --------------------------------------------------------------------------
# Words the STT commonly produces for an app name. Only applied to the word
# right after open/close/launch/start, so normal speech is untouched. Add your
# own in data/voice_aliases.json, e.g. {"ground": "chrome"}.
DEFAULT_TARGET_ALIASES = {
    "ground": "chrome", "grom": "chrome", "groom": "chrome", "crome": "chrome",
    "krome": "chrome", "krom": "chrome", "kroom": "chrome", "cro": "chrome",
    "crow": "chrome", "chrom": "chrome", "groment": "chrome",
    "spot if i": "spotify", "spotty fy": "spotify", "you tube": "youtube",
    "note pad": "notepad", "calculate": "calculator", "camra": "camera",
}
_ALIAS_FILE = Path(__file__).resolve().parents[1] / "data" / "voice_aliases.json"
_alias_cache: tuple[str, float, dict[str, str]] | None = None


def target_aliases() -> dict[str, str]:
    """Default aliases merged with data/voice_aliases.json (reloaded on change)."""
    global _alias_cache
    path = Path(os.getenv("ASTA_VOICE_ALIASES", str(_ALIAS_FILE)))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0
    if _alias_cache is not None and _alias_cache[:2] == (str(path), mtime):
        return _alias_cache[2]
    aliases = dict(DEFAULT_TARGET_ALIASES)
    if mtime >= 0:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                aliases.update(
                    {str(k).strip().lower(): str(v).strip() for k, v in loaded.items() if str(k).strip()}
                )
        except (OSError, ValueError) as exc:
            print(f"[Voice] Ignoring invalid {path.name}: {exc}", flush=True)
    _alias_cache = (str(path), mtime, aliases)
    return aliases


_TARGET_VERBS = r"(?:open|close|launch|start|switch\s+to)"


def apply_target_aliases(text: str) -> str:
    """Replace a misheard app name right after an open/close verb."""
    value = str(text or "")
    aliases = target_aliases()
    if not aliases:
        return value
    keys = sorted(aliases, key=len, reverse=True)
    pattern = re.compile(
        rf"\b({_TARGET_VERBS})\s+({'|'.join(re.escape(k) for k in keys)})\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"{m.group(1)} {aliases[m.group(2).lower()]}", value)


# "open chrome search for weather" -> "open chrome then search for weather"
_IMPLICIT_SECOND_COMMAND = re.compile(
    r"^(?P<first>(?:open|launch|start|close)\s+\S+(?:\s+\S+){0,2}?)\s+"
    r"(?P<second>(?:search(?:\s+for)?|look\s+up|open|close|launch|play)\s+\S.*)$",
    re.IGNORECASE,
)


# "and once you are there," / "then when it opens" / "and after that" between
# two commands -> a plain "and then" the router already understands.
_TRANSITION = re.compile(
    r"\s*(?:,\s*)?(?:\b(?:and|then|and\s+then)\b\s*,?\s*)?"
    r"\b(?:(?:once|when|after|as\s+soon\s+as)\s+"
    r"(?:you(?:'re|\s+are|\s+get)?|it(?:'s|\s+is)?|that(?:'s|\s+is)?|chrome|the\s+\w+)\s+"
    r"(?:there|in|open|opens|opened|loaded|loads|ready|done|up)(?:\s+there)?"
    r"|after\s+that|once\s+(?:done|open|opened|there))\b\s*,?\s*"
    r"(?=(?:search|look\s+up|open|close|launch|start|play|pause|type|go\s+to|press)\b)",
    re.IGNORECASE,
)


def strip_transitions(text: str) -> str:
    value = re.sub(
        r"^\s*(?:okay|ok|so|alright|all\s+right|now|um+|uh+)\s*[,.!]?\s+",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return _TRANSITION.sub(" and then ", value).strip()


def split_implicit_commands(text: str) -> str:
    value = " ".join(str(text or "").split())
    match = _IMPLICIT_SECOND_COMMAND.match(value)
    if not match or "," in value or re.search(r"\b(?:and|then)\b", value, re.IGNORECASE):
        return value
    return f"{match.group('first')} then {match.group('second')}"


_UNFINISHED_TAIL = re.compile(
    r"(?:^|\s)(?:open|close|launch|start|search|search\s+for|look\s+up|play|type|"
    r"and|then|and\s+then|for|to|the|a|और|फिर|ओपन|सर्च|सर्च\s+फॉर)$",
    re.IGNORECASE,
)


def ends_unfinished(text: str) -> bool:
    """True when a partial transcript stops on a verb or connector."""
    value = " ".join(words(text))
    return bool(value) and bool(_UNFINISHED_TAIL.search(value))


# --------------------------------------------------------------------------
# Search queries with Indian names / Hindi words
# --------------------------------------------------------------------------
# The English decode turns Hindi names into English look-alikes ("Hanuman
# Chalisa" -> "human challenges"), while the parallel Hindi decode usually
# gets them right ("हनुमान चलीसा"). When the Hindi query is made of known
# Hindi/Indian terms, use its romanised form instead.

HINDI_TERMS = frozenset(
    """
    hanuman chalisa bhajan bhajans aarti arti mantra mantras stotram stotra
    ram rama shri shree sri siya sita krishna krishn radha shiv shiva shankar
    mahadev ganesh ganesha ganpati durga lakshmi laxmi saraswati kali bhagwan
    bhagavan bhagwat gita geeta ramayan ramayana mahabharat katha kirtan
    gayatri sundarkand jai jay mata maa devi baba sai
    guru nanak gurbani shabad waheguru vishnu narayan om namah shivay shivaya
    hare rama bajrang bali balaji tirupati kedarnath badrinath vrindavan
    mathura ayodhya kashi banaras varanasi haridwar rishikesh
    bollywood ghazal qawwali sufi shayari dohe doha kabir tulsidas surdas
    arijit kishore lata mangeshkar rafi mukesh asha bhosle sonu nigam
    shreya ghoshal udit narayan jagjit nusrat
    biryani paneer masala dal daal chole bhature samosa pakora jalebi
    ladoo laddu halwa kheer roti paratha dosa idli sambar rasam khichdi
    rajma kadhi poha upma chai lassi
    diwali holi navratri dussehra ganeshotsav janmashtami raksha bandhan
    """.split()
)


def _simplify_roman(word: str) -> str:
    value = word.lower()
    for long, short in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u")):
        value = value.replace(long, short)
    return value


_HINDI_TERMS_SIMPLE = {
    _simplify_roman(term): term for term in sorted(HINDI_TERMS, key=lambda t: (-len(t), t))
}
_EN_QUERY = re.compile(
    r"^(?P<lead>.*?\b(?:search(?:\s+for)?|look\s+up|google|play)\s+)(?P<query>[^,.?!]+?)\s*[.?!]?$",
    re.IGNORECASE,
)
_HI_QUERY = re.compile(
    r"(?:सर्च(?:\s+(?:फॉर|फोर|फ़ॉर))?|लुक\s+अप|गूगल|प्ले)\s+(?P<query>[^,.?!।]+)$"
)


def hindi_query_terms(hindi_query: str) -> str | None:
    """Romanised Hindi query when it is made of known Hindi terms."""
    tokens = words(hindi_query)
    if not tokens or not all(_DEVANAGARI.search(t) for t in tokens):
        return None
    out, known = [], 0
    for token in tokens:
        term = _HINDI_TERMS_SIMPLE.get(_simplify_roman(romanize(token)))
        if term:
            known += 1
            out.append(term)
        else:
            out.append(DEVANAGARI_TO_ENGLISH.get(token) or _simplify_roman(romanize(token)))
    if known == 0 or known * 2 < len(tokens):
        return None
    return " ".join(out)


def repair_query_from_hindi(english: str, hindi: str) -> str:
    """Swap a garbled English search query for the Hindi decode's terms."""
    en = _EN_QUERY.match(fix_verb_typos(str(english or "").strip()))
    hi = _HI_QUERY.search(str(hindi or "").strip())
    if not en or not hi:
        return english
    query = hindi_query_terms(hi.group("query"))
    if not query:
        return english
    if query.lower() == en.group("query").strip().lower():
        return english
    english_words = set(words(en.group("query")))
    if any(term in english_words for term in query.split() if term in HINDI_TERMS):
        # The English decode already has the Hindi name.
        return english
    print(
        f"[STT] Search query from Hindi decode: {en.group('query')!r} -> {query!r}",
        flush=True,
    )
    return f"{en.group('lead')}{query}"


# Common misrecognitions of command verbs.
_VERB_TYPOS = {
    "serch": "search", "surch": "search", "sarch": "search", "saerch": "search",
    "sirch": "search", "searh": "search", "seach": "search", "sertch": "search",
    "opan": "open", "opun": "open", "opne": "open", "oppen": "open",
    "clos": "close", "klose": "close",
}
_VERB_TYPO_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_VERB_TYPOS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def fix_verb_typos(text: str) -> str:
    return _VERB_TYPO_PATTERN.sub(lambda m: _VERB_TYPOS[m.group(1).lower()], str(text or ""))
