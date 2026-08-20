"""Compare the same prompt across OpenAI and Anthropic via PromptSendingAdapter.

Anthropic is reached through its OpenAI-compatible endpoint using the same
OpenAIChatTarget class already validated in quickstart.py — no extra
dependency required.

Run from the repo root with the .env populated (see .env.example):

    python examples/compare_models.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import initialize_pyrit_async

from targets import PromptSendingAdapter, run_comparison_async

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

PROMPT = "Say hello in one sentence."


async def main() -> None:
    await initialize_pyrit_async("InMemory", env_files=[ENV_FILE])

    openai_target = OpenAIChatTarget()
    anthropic_target = OpenAIChatTarget(
        model_name=os.environ["ANTHROPIC_MODEL"],
        endpoint="https://api.anthropic.com/v1",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    adapters = {
        "openai": PromptSendingAdapter(target=openai_target, name="openai"),
        "anthropic": PromptSendingAdapter(target=anthropic_target, name="anthropic"),
    }

    comparison = await run_comparison_async(prompt=PROMPT, adapters=adapters)

    print(f"prompt: {comparison.prompt}\n")
    for name, contract in comparison.contracts.items():
        print(f"--- {name} ---")
        print(contract.final_output)
        print()


if __name__ == "__main__":
    asyncio.run(main())
