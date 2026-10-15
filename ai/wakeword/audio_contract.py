"""Shared training/inference WAV contract. Importing this module does no I/O."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import wave

WAKEWORD_SAMPLE_RATE = 16000
WAKEWORD_CHANNELS = 1
WAKEWORD_SAMPLE_WIDTH = 2


def validate_wav(path):
    with wave.open(str(path), 'rb') as audio:
        actual = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        if actual != (WAKEWORD_SAMPLE_RATE, WAKEWORD_CHANNELS, WAKEWORD_SAMPLE_WIDTH):
            raise ValueError(f'{path}: expected 16000 Hz mono 16-bit PCM, got {actual}. '
                             'Regenerate into a new directory or explicitly migrate legacy clips/manifests; do not relabel headers.')
        if audio.getnframes() <= 0:
            raise ValueError(f'{path}: empty WAV')
    return Path(path)


def normalize_wav(path):
    """Normalize a freshly generated clip, using actual resampling, not relabeling.

    Existing datasets are never traversed or rewritten automatically. Piper calls
    this only for a clip it just generated. FFmpeg is an offline-training tool.
    """
    path = Path(path)
    try:
        return validate_wav(path)
    except ValueError:
        pass
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('FFmpeg is required to resample Piper output to 16 kHz. Install FFmpeg for wakeword training.')
    descriptor, temporary = tempfile.mkstemp(prefix='.asta-wakeword-', suffix='.wav', dir=str(path.parent))
    os.close(descriptor)
    try:
        subprocess.run([executable, '-nostdin', '-y', '-v', 'error', '-i', str(path),
            '-ar', str(WAKEWORD_SAMPLE_RATE), '-ac', str(WAKEWORD_CHANNELS),
            '-c:a', 'pcm_s16le', temporary], check=True, capture_output=True, timeout=60)
        validate_wav(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
