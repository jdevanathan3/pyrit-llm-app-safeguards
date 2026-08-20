"""Run every safeguard test against both OpenAI and Anthropic, print a report.

Run from the repo root with the .env populated (see .env.example):

    python tests/run_all.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskRefusalScorer
from pyrit.setup import initialize_pyrit_async

from reporting.report import TestResult, render_markdown_summary
from scorers.injection_compliance_scorer import build_injection_compliance_scorer
from scorers.refusal_scorer import build_refusal_scorer
from targets.prompt_sending_adapter import PromptSendingAdapter
from tests import (
    test_cbrn_adversarial,
    test_direct_jailbreak,
    test_harm_baseline,
    test_indirect_injection,
    test_system_prompt_boundary,
)

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


async def main() -> None:
    await initialize_pyrit_async("InMemory", env_files=[ENV_FILE])

    openai_target = OpenAIChatTarget()
    anthropic_target = OpenAIChatTarget(
        model_name=os.environ["ANTHROPIC_MODEL"],
        endpoint="https://api.anthropic.com/v1",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    judge_target = OpenAIChatTarget()  # fixed judge for both, to keep scoring apples-to-apples

    adapters = {
        "openai": PromptSendingAdapter(target=openai_target, name="openai"),
        "anthropic": PromptSendingAdapter(target=anthropic_target, name="anthropic"),
    }

    refusal_scorer: SelfAskRefusalScorer = build_refusal_scorer(judge_target=judge_target)
    compliance_scorer = build_injection_compliance_scorer(judge_target=judge_target)

    results: list[TestResult] = []
    results.append(await test_direct_jailbreak.run(adapters=adapters, refusal_scorer=refusal_scorer))
    results.append(await test_harm_baseline.run(adapters=adapters, refusal_scorer=refusal_scorer))
    results.append(await test_cbrn_adversarial.run(adapters=adapters, refusal_scorer=refusal_scorer))
    results.append(await test_system_prompt_boundary.run(adapters=adapters))
    results.append(await test_indirect_injection.run(adapters=adapters, compliance_scorer=compliance_scorer))

    print("\n" + "=" * 72)
    print("DETAILED RESULTS")
    print("=" * 72)
    for r in results:
        print(f"\n--- {r.name} ---")
        print(r.description)
        for o in r.outcomes:
            status = "PASS" if o.passed else "FAIL"
            print(f"  [{status}] {o.model}: {o.detail}")
            print(f"      prompt:   {o.prompt_excerpt!r}")
            print(f"      response: {o.response_excerpt!r}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(render_markdown_summary(results))


if __name__ == "__main__":
    asyncio.run(main())
