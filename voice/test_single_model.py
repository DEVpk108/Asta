import sounddevice as sd
import numpy as np
from openwakeword.model import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

m = Model(
    wakeword_models=[args.model],
    inference_framework="onnx",
    vad_threshold=0,
)

print("Speak for 3 seconds...")

audio = sd.rec(
    48000,
    samplerate=16000,
    channels=1,
    dtype="int16",
    device=1,
)

sd.wait()

audio = audio[:, 0]

scores = []

for i in range(0, len(audio) - 1279, 1280):
    s = m.predict(audio[i:i+1280])
    scores.append(list(s.values())[0])

print("Scores:")
print(scores)
print()
print("MAX =", max(scores))
