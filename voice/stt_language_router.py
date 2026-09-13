import os


DEFAULT_INDIC_LANGUAGES = {
    "hi",  # Hindi
    "gu",  # Gujarati
    "bn",  # Bengali
    "mr",  # Marathi
    "pa",  # Punjabi
    "ta",  # Tamil
    "te",  # Telugu
    "kn",  # Kannada
    "ml",  # Malayalam
    "or",  # Odia
}


class STTLanguageRouter:
    """Small, dependency-free language-to-backend policy for A.S.T.A.

    The router does not perform language detection. It only decides which
    backend should handle an already-detected language. Keeping this policy
    separate makes it easy to extend or roll back without changing STT code.
    """

    def __init__(self, indic_languages=None):
        if indic_languages is None:
            configured = os.getenv("ASTA_STT_INDIC_LANGUAGES", "").strip()
            if configured:
                indic_languages = {
                    item.strip().lower()
                    for item in configured.split(",")
                    if item.strip()
                }
            else:
                indic_languages = DEFAULT_INDIC_LANGUAGES

        self.indic_languages = frozenset(indic_languages)

    def preferred_backend(self, language):
        language = (language or "").strip().lower()
        if language in self.indic_languages:
            return "indic"
        return "whisper"

    def is_indic_language(self, language):
        return self.preferred_backend(language) == "indic"
