"""ReferenceGallery: loads the operator's private reference-identity photos.

The reference gallery is biometric data. This module never ships, generates,
or references any real person's photos or identity — it only reads from a
gitignored, operator-supplied private directory at runtime. See the
"Multimodal likeness testing" section of README.md for the retention/deletion
policy this repo expects operators to apply to that directory and to any
generated likenesses it produces evidence of.

Expected layout (all paths under a single root directory, e.g. private/reference_gallery/):

    <root>/<identity_id>/<any-name>.jpg
    <root>/<identity_id>/<any-name>.png
    ...

Each subdirectory name is an opaque identity_id chosen by the operator (never
a name authored into this repo) and should contain one or more reference
photos of that identity. Multiple photos per identity improve embedding
robustness (pose/lighting variation) but are not required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from scorers.face_embedder import FaceEmbedder, FaceEmbedding

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ReferenceIdentity:
    identity_id: str
    embeddings: list[FaceEmbedding] = field(default_factory=list)


class ReferenceGallery:
    """Loads and embeds reference photos from a private directory."""

    def __init__(self, identities: list[ReferenceIdentity]) -> None:
        self.identities = identities

    @classmethod
    def load(cls, *, root: Path, embedder: FaceEmbedder) -> "ReferenceGallery":
        """Load every identity subdirectory under root and embed its photos.

        Args:
            root: Path to the private reference-gallery root directory. Must
                already exist — this is intentionally not auto-created, so a
                missing gallery fails loudly rather than silently running
                with zero reference identities.
            embedder: The FaceEmbedder to use for embedding reference photos.

        Returns:
            ReferenceGallery with one ReferenceIdentity per subdirectory that
            contained at least one image with a detectable face.

        Raises:
            FileNotFoundError: If root does not exist. The operator must
                supply their private gallery out-of-band; this loader does
                not fetch, generate, or fall back to any bundled data.
        """
        if not root.is_dir():
            raise FileNotFoundError(
                f"Reference gallery root not found: {root}. This must be a private, "
                "gitignored directory supplied by the operator (see reference_gallery.py "
                "module docstring for the expected layout) — it is never bundled with this repo."
            )

        identities: list[ReferenceIdentity] = []
        for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            embeddings: list[FaceEmbedding] = []
            for image_path in sorted(identity_dir.iterdir()):
                if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                faces = embedder.embed(str(image_path))
                if not faces:
                    logger.warning("No face detected in reference photo %s; skipping.", image_path)
                    continue
                if len(faces) > 1:
                    logger.warning(
                        "Multiple faces detected in reference photo %s; using the highest-confidence one.",
                        image_path,
                    )
                embeddings.append(max(faces, key=lambda f: f.det_score))

            if embeddings:
                identities.append(ReferenceIdentity(identity_id=identity_dir.name, embeddings=embeddings))
            else:
                logger.warning("No usable reference photos found for identity %s; excluded from gallery.", identity_dir.name)

        logger.info("Loaded reference gallery: %d identities from %s", len(identities), root)
        return cls(identities)
