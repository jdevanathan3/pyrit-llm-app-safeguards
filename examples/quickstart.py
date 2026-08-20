"""Minimal smoke test: send one prompt to OpenAIChatTarget via PromptSendingAttack.

Run from the repo root with the .env populated (see .env.example):

    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import initialize_pyrit_async

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


async def main() -> None:
    await initialize_pyrit_async("InMemory", env_files=[ENV_FILE])

    target = OpenAIChatTarget()
    attack = PromptSendingAttack(objective_target=target)

    result = await attack.execute_async(objective="Say hello in one sentence.")

    print(f"outcome: {result.outcome}")
    if result.last_response is not None:
        print(f"response: {result.last_response.converted_value}")


if __name__ == "__main__":
    asyncio.run(main())
