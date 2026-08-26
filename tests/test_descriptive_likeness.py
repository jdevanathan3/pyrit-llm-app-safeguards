"""Test: descriptive-likeness evasion suite.

Wires the image adapter + descriptive prompts + likeness scorer together.
Skips (returns None) if the operator hasn't supplied a private reference
gallery and attribute-slot file. That's by design, not a missing feature:
this suite measures real-identity-likeness risk, which requires the
operator's own private reference identities (see README's "Multimodal
likeness testing" section) — it does not run against any bundled or
synthetic placeholder identity data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pyrit.prompt_target import PromptTarget

from reporting.likeness_report import LikenessCaseRecord, LikenessRunReport
from scorers.face_embedder import InsightFaceEmbedder
from scorers.likeness_calibration import DEFAULT_CALIBRATION_PATH
from scorers.likeness_scorer import LikenessScorer
from scorers.reference_gallery import ReferenceGallery
from seed_data.descriptive_likeness_prompts import (
    DEFAULT_OPERATOR_SLOTS_PATH,
    benign_control_prompts,
    generate_descriptive_cases,
    load_operator_slots,
)
from targets.image_generation_adapter import GeminiImageClient, ImageGenerationAdapter

logger = logging.getLogger(__name__)

DEFAULT_GALLERY_ROOT = Path("private/reference_gallery")


async def run(
    *,
    gallery_root: Path = DEFAULT_GALLERY_ROOT,
    slots_path: Path = DEFAULT_OPERATOR_SLOTS_PATH,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    converter_target: PromptTarget | None = None,
) -> LikenessRunReport | None:
    """Run the descriptive-likeness suite, or skip if operator data is absent.

    Returns:
        LikenessRunReport, or None if gallery_root/slots_path don't exist
        (the expected state for anyone who hasn't configured a private
        gallery — not an error).
    """
    if not gallery_root.is_dir() or not slots_path.is_file():
        logger.warning(
            "Skipping descriptive-likeness suite: private reference gallery (%s) and/or attribute-slot "
            "file (%s) not found. Required by design — see README's Multimodal likeness testing section.",
            gallery_root,
            slots_path,
        )
        return None

    embedder = InsightFaceEmbedder()
    gallery = ReferenceGallery.load(root=gallery_root, embedder=embedder)
    scorer = LikenessScorer(embedder=embedder, gallery=gallery, calibration_path=calibration_path)

    image_client = GeminiImageClient(api_key=os.environ["GEMINI_API_KEY"])
    adapter = ImageGenerationAdapter(client=image_client, name="gemini-image")

    slots_by_identity = load_operator_slots(slots_path)
    attack_cases = await generate_descriptive_cases(slots_by_identity=slots_by_identity, converter_target=converter_target)

    report = LikenessRunReport()

    for case in attack_cases:
        contract = await adapter.run_turn(case.prompt)
        likeness_results = scorer.score_contract(contract) if contract.outcome == "generated" else []
        report.records.append(
            LikenessCaseRecord(
                prompt=case.prompt,
                target_identity_id=case.target_identity_id,
                attribute_pattern=slots_by_identity[case.target_identity_id].role,
                contract_outcome=contract.outcome,
                likeness_results=likeness_results,
            )
        )

    for prompt in benign_control_prompts():
        contract = await adapter.run_turn(prompt)
        likeness_results = scorer.score_contract(contract) if contract.outcome == "generated" else []
        report.records.append(
            LikenessCaseRecord(
                prompt=prompt,
                target_identity_id=None,
                attribute_pattern="benign",
                contract_outcome=contract.outcome,
                likeness_results=likeness_results,
            )
        )

    return report
