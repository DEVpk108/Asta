"""Opt-in camera experiment, not identity authorization or a kernel component.

Use only a local numeric .npy embedding array created from trusted enrollment.
Legacy pickle artifacts are deliberately not loaded or migrated automatically.
"""
import os
from pathlib import Path


class FaceRecognizer:
    def __init__(self, encodings_path=None):
        self.encodings_path = Path(encodings_path or os.getenv('ASTA_FACE_ENCODINGS_PATH')
                                   or Path.home()/'.asta'/'face_encodings.npy').expanduser()

    def face_reco(self, *, max_frames=300):
        import cv2
        import face_recognition
        import numpy as np
        known = np.load(self.encodings_path, allow_pickle=False)
        if known.dtype.kind not in 'fi' or known.ndim != 2 or known.shape[1] != 128 or not np.isfinite(known).all():
            raise ValueError('Expected a finite numeric (N, 128) face embedding array')
        camera = cv2.VideoCapture(0)
        try:
            if not camera.isOpened():
                return False
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(max(1, int(max_frames))):
                ok, frame = camera.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb)
                for encoding in face_recognition.face_encodings(rgb, locations):
                    if any(face_recognition.compare_faces(known, encoding)):
                        return True
            return False
        finally:
            camera.release()


if __name__ == '__main__':
    print(FaceRecognizer().face_reco())
