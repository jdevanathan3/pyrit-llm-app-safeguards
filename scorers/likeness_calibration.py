"""Threshold calibration and detector-bias validation for the likeness scorer.

Both routines take a labeled validation set supplied by the operator — this
module ships no validation data and no threshold. A hardcoded "magic number"
similarity threshold is exactly what Section 4 of the implementation brief
prohibits: the threshold this module produces is written to a config file
(likeness_calibration.json, gitignored) for likeness_scorer.py to load, not
baked into code.

Labeled validation set format (LabeledCase): each case is a probe image plus
ground truth of whether it genuinely is one of the gallery's reference
identities, plus an operator-supplied demographic slice label. The
demographic labels must come from the operator's validation-set metadata —
never from the face detector's own age/sex predictions, which would make
the bias check circular (validating a detector using the same detector's
guesses about the people in the validation set).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scorers.face_embedder import FaceEmbedder, cosine_similarity
from scorers.reference_gallery import ReferenceGallery

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_PATH = Path("private/likeness_calibration.json")


@dataclass
class LabeledCase:
    """One validation-set entry.

    Attributes:
        image_path: Path to the probe image (operator-supplied; not committed).
        true_identity_id: The gallery identity_id this probe genuinely
            depicts, or None if the probe is a non-match (a real person who
            is not any reference identity, or a generic/synthetic face).
        demographic_slice: Operator-labeled demographic bucket for bias
            reporting (e.g. "skin_tone:dark", "age:60+"). Never inferred from
            the detector.
    """

    image_path: str
    true_identity_id: str | None
    demographic_slice: str | None = None


@dataclass
class _ScoredCase:
    case: LabeledCase
    similarity: float
    is_genuine: bool


def _score_cases(*, gallery: ReferenceGallery, embedder: FaceEmbedder, cases: list[LabeledCase]) -> list[_ScoredCase]:
    by_id = {identity.identity_id: identity for identity in gallery.identities}
    scored: list[_ScoredCase] = []
    for case in cases:
        faces = embedder.embed(case.image_path)
        if not faces:
            logger.warning("No face detected in validation case %s; excluded from calibration.", case.image_path)
            continue
        probe = max(faces, key=lambda f: f.det_score).vector

        if case.true_identity_id is not None:
            identity = by_id.get(case.true_identity_id)
            if identity is None:
                raise ValueError(f"Labeled case references unknown identity_id: {case.true_identity_id}")
            similarity = max(cosine_similarity(probe, ref.vector) for ref in identity.embeddings)
            is_genuine = True
        else:
            # Non-match case: worst-case (highest) similarity against ANY gallery identity.
            similarity = max(
                (cosine_similarity(probe, ref.vector) for identity in gallery.identities for ref in identity.embeddings),
                default=0.0,
            )
            is_genuine = False

        scored.append(_ScoredCase(case=case, similarity=similarity, is_genuine=is_genuine))
    return scored


@dataclass
class CalibrationResult:
    threshold: float
    true_positive_rate: float
    false_positive_rate: float
    n_genuine: int
    n_impostor: int


def calibrate_threshold(
    *,
    gallery: ReferenceGallery,
    embedder: FaceEmbedder,
    labeled_cases: list[LabeledCase],
    target_false_positive_rate: float | None = None,
) -> CalibrationResult:
    """Sweep candidate thresholds against a labeled validation set (ROC-style).

    Args:
        gallery: The reference gallery to validate against.
        embedder: The FaceEmbedder used to embed probe images.
        labeled_cases: Ground-truth genuine/impostor probes. Needs both
            classes represented to produce a meaningful threshold.
        target_false_positive_rate: If given, pick the lowest threshold that
            achieves at most this FPR. If None, maximize Youden's J
            statistic (TPR - FPR), a standard threshold-free-of-class-balance
            choice.

    Returns:
        CalibrationResult with the chosen threshold and its TPR/FPR on this
        validation set. Save it with save_calibration() before using it in
        LikenessScorer — this function does not persist anything itself.
    """
    scored = _score_cases(gallery=gallery, embedder=embedder, cases=labeled_cases)
    genuine = [s.similarity for s in scored if s.is_genuine]
    impostor = [s.similarity for s in scored if not s.is_genuine]
    if not genuine or not impostor:
        raise ValueError(
            f"Calibration requires both genuine and impostor cases with detectable faces; "
            f"got {len(genuine)} genuine, {len(impostor)} impostor."
        )

    candidate_thresholds = sorted(set(genuine) | set(impostor))
    best: CalibrationResult | None = None
    for t in candidate_thresholds:
        tpr = sum(1 for s in genuine if s >= t) / len(genuine)
        fpr = sum(1 for s in impostor if s >= t) / len(impostor)
        if target_false_positive_rate is not None:
            if fpr <= target_false_positive_rate and (best is None or t < best.threshold):
                best = CalibrationResult(
                    threshold=t, true_positive_rate=tpr, false_positive_rate=fpr, n_genuine=len(genuine), n_impostor=len(impostor)
                )
        else:
            j = tpr - fpr
            if best is None or j > (best.true_positive_rate - best.false_positive_rate):
                best = CalibrationResult(
                    threshold=t, true_positive_rate=tpr, false_positive_rate=fpr, n_genuine=len(genuine), n_impostor=len(impostor)
                )

    if best is None:
        raise ValueError(f"No threshold achieved target_false_positive_rate={target_false_positive_rate}")
    return best


@dataclass
class DemographicSliceReport:
    slice_name: str
    n: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float


def validate_detector_bias(
    *, gallery: ReferenceGallery, embedder: FaceEmbedder, threshold: float, labeled_cases: list[LabeledCase]
) -> list[DemographicSliceReport]:
    """Report detector accuracy sliced by operator-labeled demographic group.

    Mandatory per the implementation brief: an uncalibrated detector, or one
    validated only on aggregate accuracy, can hide large per-group error-rate
    gaps. Cases without a demographic_slice label are grouped under
    "unlabeled" and still reported, but a majority-unlabeled validation set
    should be treated as itself a finding — it means this check hasn't
    actually been done.

    Returns:
        One DemographicSliceReport per distinct slice label present in
        labeled_cases (including "unlabeled").
    """
    scored = _score_cases(gallery=gallery, embedder=embedder, cases=labeled_cases)

    slices: dict[str, list[_ScoredCase]] = {}
    for s in scored:
        key = s.case.demographic_slice or "unlabeled"
        slices.setdefault(key, []).append(s)

    reports = []
    for slice_name, items in sorted(slices.items()):
        correct = 0
        false_positives = 0
        false_negatives = 0
        n_genuine = sum(1 for i in items if i.is_genuine)
        n_impostor = len(items) - n_genuine
        for item in items:
            predicted_match = item.similarity >= threshold
            if predicted_match == item.is_genuine:
                correct += 1
            elif predicted_match and not item.is_genuine:
                false_positives += 1
            elif not predicted_match and item.is_genuine:
                false_negatives += 1
        reports.append(
            DemographicSliceReport(
                slice_name=slice_name,
                n=len(items),
                accuracy=correct / len(items) if items else 0.0,
                false_positive_rate=false_positives / n_impostor if n_impostor else 0.0,
                false_negative_rate=false_negatives / n_genuine if n_genuine else 0.0,
            )
        )

    if "unlabeled" in slices and len(slices["unlabeled"]) > len(scored) / 2:
        logger.warning(
            "Detector-bias validation: majority of validation cases (%d/%d) have no demographic_slice "
            "label. This bias check is not meaningful without labeled slices — supply demographic "
            "metadata in the validation set.",
            len(slices["unlabeled"]),
            len(scored),
        )

    return reports


def save_calibration(
    *,
    path: Path = DEFAULT_CALIBRATION_PATH,
    result: CalibrationResult,
    bias_report: list[DemographicSliceReport] | None = None,
) -> None:
    """Persist a calibrated threshold (and optional bias report) to config.

    Writes JSON, not code — per the brief, the threshold must live in config.
    path defaults under private/ (gitignored); pass an explicit path to store
    it elsewhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": result.threshold,
        "true_positive_rate": result.true_positive_rate,
        "false_positive_rate": result.false_positive_rate,
        "n_genuine": result.n_genuine,
        "n_impostor": result.n_impostor,
        "bias_report": [vars(r) for r in bias_report] if bias_report else None,
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved likeness calibration to %s (threshold=%.4f)", path, result.threshold)


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> float:
    """Load a previously calibrated threshold from config.

    Raises:
        FileNotFoundError: if path doesn't exist — callers should treat this
            as "uncalibrated" per likeness_scorer.py's handling, not silently
            substitute a default.
    """
    payload = json.loads(path.read_text())
    return float(payload["threshold"])
