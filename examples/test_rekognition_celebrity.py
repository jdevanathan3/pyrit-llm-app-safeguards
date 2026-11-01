"""Generate a celebrity image with Gemini and check whether AWS Rekognition recognizes it.

Run from the repo root with .env populated (GEMINI_API_KEY plus standard AWS
credentials with rekognition:RecognizeCelebrities permission):

    python examples/test_rekognition_celebrity.py
    python examples/test_rekognition_celebrity.py --celebrity "Robert De Niro" --model gemini-2.5-flash-image

To check Rekognition alone against an image you already have (no Gemini call):

    python examples/test_rekognition_celebrity.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from pyrit.setup import initialize_pyrit_async

from scorers.rekognition_celebrity_scorer import DEFAULT_MIN_MATCH_CONFIDENCE, RekognitionCelebrityScorer
from targets.image_generation_adapter import NANO_BANANA_PRO, GeminiImageClient, ImageGenerationAdapter
from tests import test_rekognition_celebrity

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--celebrity", default="Robert De Niro", help="Name Rekognition is expected to return.")
    parser.add_argument("--model", default=NANO_BANANA_PRO, help="Gemini image model id.")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_MATCH_CONFIDENCE)
    parser.add_argument("--image", help="Skip generation; score this existing image with Rekognition only.")
    args = parser.parse_args()

    # InMemory: the scorer writes Score rows to PyRIT's CentralMemory.
    await initialize_pyrit_async("InMemory", env_files=[ENV_FILE] if ENV_FILE.exists() else None)

    scorer = RekognitionCelebrityScorer(expected_celebrity=args.celebrity, min_match_confidence=args.min_confidence)

    if args.image:
        for score in await scorer.score_image_async(args.image):
            print(f"detected={score.get_value()}  {score.score_rationale}")
        return

    adapter = ImageGenerationAdapter(
        client=GeminiImageClient(api_key=os.environ["GEMINI_API_KEY"], model_id=args.model),
        name=f"gemini:{args.model}",
    )
    report = await test_rekognition_celebrity.run(
        adapter=adapter, scorer=scorer, celebrity=args.celebrity, generator_name=adapter.name
    )
    print(report.render_markdown())
    for case in report.cases:
        for path in case.image_paths:
            print(f"[{case.kind}/{case.result}] image saved: {path}")


if __name__ == "__main__":
    asyncio.run(main())
