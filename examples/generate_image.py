"""Minimal smoke test: send one prompt to GeminiImageClient via ImageGenerationAdapter.

Saves the generated image under outputs/generated/ (gitignored) and prints
its path plus the outcome. Doesn't need initialize_pyrit_async — unlike the
chat adapters, GeminiImageClient wraps the google-genai SDK directly and
never touches PyRIT's CentralMemory.

Run from the repo root with the .env populated (see .env.example):

    python examples/generate_image.py
    python examples/generate_image.py "a lighthouse at sunset, watercolor style"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import dotenv

from targets.image_generation_adapter import GeminiImageClient, ImageGenerationAdapter

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_PROMPT = "A golden retriever sitting in a sunny park, photorealistic."


async def main() -> None:
    dotenv.load_dotenv(ENV_FILE)

    import os

    client = GeminiImageClient(api_key=os.environ["GEMINI_API_KEY"])
    adapter = ImageGenerationAdapter(client=client, name="gemini-image")

    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    print(f"model: {client.model_id}")
    print(f"prompt: {prompt}")

    contract = await adapter.run_turn(prompt)

    print(f"outcome: {contract.outcome}")
    if contract.final_output:
        print(f"text: {contract.final_output}")
    for image in contract.generated_images:
        print(f"saved image: {image.path} ({image.mime_type})")

    if contract.outcome == "generated" and contract.generated_images:
        print(f"\nView it with: open {contract.generated_images[0].path}")


if __name__ == "__main__":
    asyncio.run(main())
