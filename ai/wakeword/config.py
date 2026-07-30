from pathlib import Path

# Root of wakeword module
ROOT = Path(__file__).resolve().parent

# Dataset
DATASET_DIR = ROOT / "dataset" / "v1"

POSITIVE_DIR = DATASET_DIR / "positive"
NEGATIVE_DIR = DATASET_DIR / "negative_wav"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# Generated data
GENERATED_DIR = ROOT / "generated"

# Models
MODEL_DIR = ROOT / "models"

# OpenWakeWord
OWW_DIR = ROOT / "openWakeWord"

# Temporary cache
CACHE_DIR = ROOT / "cache"


####################################################
# Wakeword Variants
####################################################

WAKEWORD_VARIANTS = {

    "Hey ASTA": [

        {
            "text": "Hey AASTA",
            "weight": 50,
        },

        {
            "text": "Hey AASTA.",
            "weight": 10,
        },

        {
            "text": "Hey AASTA!",
            "weight": 8,
        },

        {
            "text": "Hey AASTA...",
            "weight": 6,
        },

        {
            "text": "...Hey AASTA",
            "weight": 4,
        },

        {
            "text": "Hey, AASTA",
            "weight": 7,
        },

        {
            "text": "Hey AASTA?",
            "weight": 5,
        },

        {
            "text": "HEY AASTA",
            "weight": 5,
        },

        {
            "text": "Hey aasta",
            "weight": 3,
        },

        {
            "text": "hey AASTA",
            "weight": 2,
        },

    ],

    "Hello ASTA": [

        {
            "text": "Hello AASTA",
            "weight": 30,
        },

        {
            "text": "Hello AASTA.",
            "weight": 10,
        },

        {
            "text": "Hello AASTA!",
            "weight": 8,
        },

        {
            "text": "Hello, AASTA",
            "weight": 7,
        },

        {
            "text": "...Hello AASTA",
            "weight": 5,
        },

    ],

    "Wake up ASTA": [

        {
            "text": "Wake up AASTA",
            "weight": 25,
        },

        {
            "text": "Wake up AASTA!",
            "weight": 10,
        },

        {
            "text": "Wake up, AASTA",
            "weight": 8,
        },

        {
            "text": "...Wake up AASTA",
            "weight": 5,
        },

        {
            "text": "Wake up AASTA.",
            "weight": 7,
        },

    ]

}


SAMPLES_PER_PHRASE = 1000

PIPER_DIR = ROOT / "tools" / "piper"

PIPER_EXECUTABLE = PIPER_DIR / "piper.exe"

PIPER_MODEL = (
    PIPER_DIR
    / "voices"
    / "en_US-l2arctic-medium.onnx"
)

PIPER_CONFIG = (
    PIPER_DIR
    / "voices"
    / "en_US-l2arctic-medium.onnx.json"
)


SPEAKER_WEIGHTS = {

    0: 15,
    1: 10,
    2: 5,
    3: 20,
    4: 12,
    5: 18,
    6: 3,
    7: 15,

}