"""FaceEmbedder: pluggable face-detection + embedding interface.

Kept behind an ABC so the operator can swap the underlying face-recognition
model without touching likeness_scorer.py or reference_gallery.py. The
default implementation (InsightFaceEmbedder) uses insightface's buffalo_l
model (ArcFace embeddings) via onnxruntime — see MULTIMODAL_API_NOTES.md for
why this was chosen over face_recognition/dlib (no native compilation needed
for this project's Python/platform) and dependency notes (first use
downloads the ~350MB model bundle; needs network access once).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceEmbedding:
    """One detected face and its embedding vector.

    Attributes:
        vector: The face-embedding vector (not assumed normalized by callers
            — similarity functions normalize explicitly).
        bbox: (x1, y1, x2, y2) pixel bounding box of the detected face.
        det_score: The detector's own confidence for this face, 0-1.
    """

    vector: np.ndarray
    bbox: tuple[int, int, int, int]
    det_score: float


class FaceEmbedder(ABC):
    """Detects faces in an image and returns an embedding per face."""

    @abstractmethod
    def embed(self, image_path: str) -> list[FaceEmbedding]:
        """Detect every face in the image at image_path and embed each one.

        Args:
            image_path: Local path to the image file.

        Returns:
            list[FaceEmbedding]: One entry per detected face, empty if none
            detected. Callers must handle the empty case (no face) distinctly
            from a low-similarity match — a generated image with no
            detectable face is not evidence either way.
        """


class InsightFaceEmbedder(FaceEmbedder):
    """Default FaceEmbedder using insightface's buffalo_l (ArcFace) model."""

    def __init__(self, *, det_thresh: float = 0.5) -> None:
        from insightface.app import FaceAnalysis  # local import: optional heavy dependency

        self._app = FaceAnalysis(name="buffalo_l")
        self._app.prepare(ctx_id=-1, det_thresh=det_thresh)  # ctx_id=-1: CPU

    def embed(self, image_path: str) -> list[FaceEmbedding]:
        import cv2  # local import: optional dependency, matches insightface's own expectation of BGR ndarrays

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self._app.get(image)
        return [
            FaceEmbedding(
                vector=face.embedding,
                bbox=tuple(int(v) for v in face.bbox),
                det_score=float(face.det_score),
            )
            for face in faces
        ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors, in [-1, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
