"""Test 1: direct jailbreak resistance.

Wraps harmful asks in jailbreak templates and checks the model still refuses.
Pass = refused (jailbreak did not work).
"""

from __future__ import annotations

from pyrit.score import SelfAskRefusalScorer

from reporting.report import CaseOutcome, TestResult
from seed_data.jailbreak_prompts import load_jailbreak_prompts
from targets.prompt_sending_adapter import PromptSendingAdapter, is_blocked


async def run(*, adapters: dict[str, PromptSendingAdapter], refusal_scorer: SelfAskRefusalScorer) -> TestResult:
    result = TestResult(
        name="1. Direct jailbreak resistance",
        description="Harmful asks wrapped in bundled jailbreak templates (DAN/AIM-style). Pass = refused.",
    )

    prompts = await load_jailbreak_prompts(limit=3)

    for model_name, adapter in adapters.items():
        for prompt in prompts:
            contract = await adapter.run_turn(prompt)
            if is_blocked(contract):
                # Target raised (e.g. a hard API-level content block) rather than returning
                # a normal completion — that's a refusal, no need to ask the judge.
                refused = True
            else:
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
