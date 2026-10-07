"""Measure A.S.T.A. speech latency on this machine.

    python scripts/benchmark_speech.py                 # record 5 s from the mic
    python scripts/benchmark_speech.py --wav me.wav    # use a 16 kHz mono WAV
    python scripts/benchmark_speech.py --skip-whisper

Reports, for the same audio:
  * Nemotron streaming: delay from end of speech to final transcript
    (audio is fed at real-time pace, like a live microphone)
  * Whisper (ASTA_STT_MODEL, default medium): full re-decode time
  * Kokoro: time to first audio for an English and a Hindi sentence
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as handle:
        if handle.getframerate() != 16000 or handle.getnchannels() != 1:
            raise SystemExit("Use a 16 kHz mono WAV (ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd

    device = sd.query_devices(kind="input")
    print(f"Microphone: {device['name']}")
    for count in (3, 2, 1):
        print(f"  starting in {count}...", flush=True)
        time.sleep(1)
    print(f"SPEAK NOW for {seconds:.0f} s (English or Hindi)...", flush=True)
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    audio = audio.reshape(-1)
    saved = ROOT / "models" / "speech" / "benchmark.wav"
    saved.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(saved), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"Saved recording: {saved} (rerun with --wav to compare on the same audio)")
    return audio


def bench_nemotron(audio: np.ndarray) -> None:
    from voice.nemotron_streaming_engine import NemotronStreamingEngine

    engine = NemotronStreamingEngine()
    stream = engine.create_stream()
    start = time.perf_counter()
    for index in range(0, audio.size, 512):
        stream.accept(audio[index : index + 512])
        target = start + (index + 512) / 16000
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    end_of_audio = time.perf_counter()
    text = stream.finish()
    language = getattr(stream, "language", None) or engine.language
    print(
        f"Nemotron  end-of-speech -> text: {time.perf_counter() - end_of_audio:.3f}s  "
        f"[{language}] {text!r}"
    )


def bench_whisper(audio: np.ndarray) -> None:
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = os.getenv("ASTA_STT_MODEL", "medium")
    model = WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else "int8")
    list(model.transcribe(np.zeros(16000, np.float32), beam_size=1)[0])  # warm up
    start = time.perf_counter()
    segments, info = model.transcribe(audio, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)
    print(
        f"Whisper   {name}/{device} end-of-speech -> text: "
        f"{time.perf_counter() - start:.3f}s  [{info.language}] {text!r}"
    )


def bench_kokoro() -> None:
    from speech.kokoro_engine import KokoroEngine

    engine = KokoroEngine()
    for label, text in (
        ("English", "I opened Chrome for you, and the weather today looks sunny with a light breeze."),
        ("Hindi", "मैंने आपके लिए क्रोम खोल दिया है, और आज मौसम बहुत अच्छा है।"),
    ):
        start = time.perf_counter()
        first = None
        for _chunk in engine.iter_audio(text):
            if first is None:
                first = time.perf_counter() - start
        print(f"Kokoro    {label} time to first audio: {first if first is not None else float('nan'):.3f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-kokoro", action="store_true")
    args = parser.parse_args()

    audio = load_wav(args.wav) if args.wav else record(args.seconds)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    print(f"Audio: {audio.size / 16000:.1f}s  peak={peak:.3f}  rms={rms:.4f}")
    if peak < 0.04:
        print(
            "WARNING: the recording is nearly silent. Check the Windows default "
            "input device / mic volume (A.S.T.A. uses the same device)."
        )
    print()
    bench_nemotron(audio)
    if not args.skip_whisper:
        bench_whisper(audio)
    if not args.skip_kokoro:
        bench_kokoro()
    return 0


if __name__ == "__main__":
    sys.exit(main())
