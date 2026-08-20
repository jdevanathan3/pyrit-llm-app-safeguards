"""Test 4: system-prompt boundary / extraction resistance.

Plants a canary string in a system prompt and tries to extract it back.
Pass = canary did not leak into final_output. Deterministic (no LLM judge).
"""

from __future__ import annotations

from pyrit.models import Message

from reporting.report import CaseOutcome, TestResult
from scorers.canary_leak_scorer import check_canary_leak
from seed_data.system_prompt_extraction_prompts import CANARY, EXTRACTION_PROMPTS, SYSTEM_PROMPT
from targets.prompt_sending_adapter import PromptSendingAdapter


async def run(*, adapters: dict[str, PromptSendingAdapter]) -> TestResult:
    result = TestResult(
        name="4. System-prompt boundary / extraction resistance",
        description="Canary planted in system prompt; pass = it did not leak into the response.",
    )

    system_message = Message.from_prompt(prompt=SYSTEM_PROMPT, role="system")

    for model_name, adapter in adapters.items():
        for prompt in EXTRACTION_PROMPTS:
            contract = await adapter.run_turn(prompt, prepended_conversation=[system_message])
            leaked, index = check_canary_leak(contract=contract, canary=CANARY)
            result.outcomes.append(
                CaseOutcome(
                    model=model_name,
                    passed=not leaked,
                    prompt_excerpt=prompt[:120],
                    response_excerpt=contract.final_output[:200],
                    detail="LEAKED at index " + str(index) if leaked else "boundary held",
                )
            )

    return result
