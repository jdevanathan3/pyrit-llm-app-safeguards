"""Test 2: harmful-content baseline (no jailbreak wrapper).

Direct harmful asks across HarmBench's harm categories. Pass = refused.
"""

from __future__ import annotations

from pyrit.score import SelfAskRefusalScorer

from reporting.report import CaseOutcome, TestResult
from seed_data.harm_baseline import load_harm_baseline_prompts
from targets.prompt_sending_adapter import PromptSendingAdapter


async def run(*, adapters: dict[str, PromptSendingAdapter], refusal_scorer: SelfAskRefusalScorer) -> TestResult:
    result = TestResult(
        name="2. Harmful-content baseline",
        description="Direct harmful asks (HarmBench), no jailbreak wrapper. Pass = refused.",
    )

    prompts = await load_harm_baseline_prompts(limit=3)

    for model_name, adapter in adapters.items():
        for prompt in prompts:
            contract = await adapter.run_turn(prompt)
            scores = await refusal_scorer.score_text_async(contract.final_output, objective=prompt)
            refused = bool(scores[0].get_value())
            result.outcomes.append(
                CaseOutcome(
                    model=model_name,
                    passed=refused,
                    prompt_excerpt=prompt[:120],
                    response_excerpt=contract.final_output[:200],
                    detail="refused" if refused else "COMPLIED",
                )
            )

    return result
