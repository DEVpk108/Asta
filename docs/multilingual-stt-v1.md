# A.S.T.A. Multilingual STT v1

This change is intentionally isolated behind the `multilingual` STT backend.
The existing `whisper`, `indic`, and `hybrid` backends remain unchanged in their
selection behavior.

## Enable

Set:

```text
ASTA_STT_BACKEND=multilingual
```

The multilingual path asks faster-whisper to detect the language automatically.
When the detected language is one of the configured Indic languages and the
language probability is at least `0.70`, A.S.T.A. tries IndicConformer and falls
back to Whisper if IndicConformer is unavailable or produces an unusable result.

The Indic language set can be overridden with a comma-separated value:

```text
ASTA_STT_INDIC_LANGUAGES=hi,gu,bn,mr,pa,ta,te,kn,ml,or
```

## Hindi test

Speak a sentence such as:

```text
Asta mera project folder kholo
```

Expected routing in the logs:

```text
[STT] Detected Indic language=hi prob=...
[STT] indic-conformer language=hi ...
```

The final transcript is still returned through the existing `RecognitionEngine`
string interface, so the rest of the VoiceModule does not need to change.

## Rollback

Remove or unset `ASTA_STT_BACKEND=multilingual` and use the previous backend,
for example:

```text
ASTA_STT_BACKEND=whisper
```

To fully discard the experimental implementation, close/delete the feature PR
and continue using `fix/voice-tool-execution`. No stable-branch code needs to be
reverted.
