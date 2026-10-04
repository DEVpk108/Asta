"""Pick between parallel English and Hindi transcripts of the same audio.

Nemotron's automatic language ID tends to write Indian-accented English in
Devanagari ("ओपन क्रोम" for "open Chrome"), and per-token confidence is not
comparable across scripts. Real English transcripts are dense with common
English words; real Hindi transcripts are dense with Hindi function words,
pronouns and verbs, while English-sounding speech decoded as Hindi is not.
"""

from __future__ import annotations

import unicodedata

from core.transliteration import (
    DEVANAGARI_TO_ENGLISH,
    devanagari_english,
    repair_query_from_hindi,
    strip_wake_remnant,
)

ENGLISH_COMMON = frozenset(
    """
    a about above after again all also am an and any are as ask at back be because
    been before being below between both but by can cancel check close come could
    day did do does doing done down each email every file find first for from get
    give go going good got had has have he hello her here hey hi him his how i if
    in into is it its just know last let like look make me message more most music
    my name need new next no not now of off ok okay on one only open or other our
    out over pause play please put remind remember résumé right run say search see
    send set she should show so some start stop take tell than thank thanks that
    the their them then there these they thing this those time to today tomorrow
    turn two up us use very volume want was we weather were what when where which
    while who why will with would write yes yesterday you your yours
    app apps browser chrome code computer desktop document download folder google
    joke minute minutes mute news note notepad notes reminder screen screenshot
    song songs spotify timer track video window youtube increase decrease lower
    louder quieter brightness alarm calendar meeting shut down restart lock
    morning evening night week weekend hour hours latest current back forward
    explain summarize translate question answer help
    """.split()
)

HINDI_COMMON = frozenset(
    """
    है हैं हूँ हूं हो था थी थे थीं गया गयी गई गए रहा रही रहे सकता सकती सकते
    का की के को में से पर तक ने लिए साथ बारे बाद पहले अंदर बाहर ऊपर नीचे
    और या लेकिन पर मगर तो भी ही नहीं ना मत न हाँ हां जी अगर क्योंकि जब तब
    मैं मेरा मेरी मेरे मुझे मुझको हम हमारा हमारी हमारे हमें तुम तुम्हारा तुम्हें आप
    आपका आपकी आपके आपको वह वो वे उस उसका उसकी उसे उन उनका उन्हें यह ये इस इसका
    इसे इन कोई कुछ सब सभी क्या क्यों कैसे कैसा कैसी कहाँ कहां कब कौन कितना कितने
    कितनी जो जिस जिसे वाला वाली वाले
    आज कल अभी अब फिर जल्दी बहुत थोड़ा ज़्यादा ज्यादा कम अच्छा अच्छी अच्छे ठीक
    खोलो खोल खोलना खोलें बंद करो करें करना कर करके करते करती दो दें देना दीजिए
    बताओ बताइए बताना बता सुनाओ सुनाइए सुनो चलाओ चला चलाना बजाओ बजा लगाओ लगा
    दिखाओ दिखा भेजो भेज लिखो लिख ढूंढो ढूँढो खोजो याद रखो रोको रोक बढ़ाओ बढ़ा
    घटाओ घटा ज़रा जरा धन्यवाद शुक्रिया नमस्ते
    मौसम समय गाना गाने आवाज़ आवाज हालचाल हाल चाल बात दिन रात सुबह शाम
    नाम काम घर दोस्त चुटकुला सवाल जवाब मदद
    """.split()
)

def _words(text: str) -> list[str]:
    # Keep letters and combining vowel signs (Devanagari matras are category
    # M, which \w-based regexes treat as punctuation).
    value = "".join(
        ch if unicodedata.category(ch)[0] in "LM" else " "
        for ch in str(text or "").lower()
    )
    return value.split()


# Words that make an English command recognisable (common words plus the
# command/app vocabulary the Hindi decode transliterates).
_ENGLISH_KNOWN = ENGLISH_COMMON | frozenset(DEVANAGARI_TO_ENGLISH.values())

# Share of Devanagari words that must be transliterated English before the
# Hindi decode is treated as English speech.
TRANSLITERATED_ENGLISH_COVERAGE = 0.6


def _known_english_words(text: str) -> int:
    return sum(1 for word in _words(text) if word in _ENGLISH_KNOWN)


def english_score(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    return sum(1 for word in words if word in ENGLISH_COMMON) / len(words)


def hindi_score(text: str) -> float:
    words = _words(text)
    if not words:
        return 0.0
    return sum(1 for word in words if word in HINDI_COMMON) / len(words)


def choose_transcript(*args, **kwargs) -> tuple[str, str]:
    """Return ``(language, text)``; English searches borrow Hindi names."""
    language, text = _choose_transcript(*args, **kwargs)
    if language == "en" and len(args) >= 2:
        text = repair_query_from_hindi(text, strip_wake_remnant(str(args[1] or "")))
    return language, text


def _choose_transcript(
    english: str,
    hindi: str,
    *,
    english_confidence: float | None = None,
    hindi_confidence: float | None = None,
) -> tuple[str, str]:
    """Return ``(language, text)`` for the more plausible transcript."""
    english = strip_wake_remnant(str(english or "").strip())
    hindi = strip_wake_remnant(str(hindi or "").strip())
    if not hindi:
        return "en", english

    # Indian-accented English often decodes as Devanagari-English in the
    # Hindi stream ("ओपन क्रोम") while the English stream clips or garbles
    # it ("Open", "Upnro"). Map it back and keep whichever English reading
    # has more recognisable words; ties keep the English decode.
    converted, coverage = devanagari_english(
        hindi, fuzzy=True, exclude=HINDI_COMMON
    )
    if converted and coverage >= TRANSLITERATED_ENGLISH_COVERAGE:
        if not english:
            return "en", converted
        if _known_english_words(converted) > _known_english_words(english):
            return "en", converted
        return "en", english

    if not english:
        return "hi", hindi

    en_score = english_score(english)
    hi_score = hindi_score(hindi)
    if hi_score > en_score or (hi_score == en_score and hi_score > 0):
        # Ties with real Hindi words present (Hinglish such as
        # "Chrome खोलो और music बजाओ") are Hindi sentences.
        return "hi", hindi
    if en_score > hi_score:
        return "en", english
    # Neither looks like ordinary speech (names, single words): fall back to
    # the model's own confidence, then to English for the command router.
    if (
        english_confidence is not None
        and hindi_confidence is not None
        and hindi_confidence > english_confidence + 0.05
    ):
        return "hi", hindi
    return "en", english
