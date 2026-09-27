"""A.S.T.A. Vision Package.

Vision backends are loaded lazily so lightweight capabilities such as
screenshots do not require the optional face-recognition stack at import time.
"""

__all__ = [
    "FaceRecognition",
    "capture_screenshot",
    "LFM25VLEngine",
    "VisionEngineError",
]


def __getattr__(name):
    if name == "LFM25VLEngine":
        from .lfm2_5_vl_engine import LFM25VLEngine
        return LFM25VLEngine
    if name == "VisionEngineError":
        from .lfm2_5_vl_engine import VisionEngineError
        return VisionEngineError
    if name == "FaceRecognition":
        from .face_recognition import FaceRecognizer

        return FaceRecognizer
    if name == "capture_screenshot":
        from .screenshot_backend import capture_screenshot

        return capture_screenshot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
