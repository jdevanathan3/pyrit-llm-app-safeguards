"""Seed prompts for Test 1: direct jailbreak resistance.

Wraps HarmBench behaviors in PyRIT's bundled jailbreak templates (DAN, AIM,
etc.) — the "user is the adversary" threat model from Anthropic's guardrails
guidance: https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks
"""

from __future__ import annotations

from pyrit.datasets import TextJailBreak

from seed_data.harmbench import load_harmbench_behaviors


async def load_jailbreak_prompts(*, limit: int = 3) -> list[str]:
    """Build jailbreak-wrapped harmful prompts.

    Args:
        limit: Number of HarmBench behaviors to wrap.

    Returns:
        list[str]: Jailbreak-template-wrapped prompts, ready to send to a target.
    """
    behaviors = await load_harmbench_behaviors(limit=limit)
    jailbreak = TextJailBreak(random_template=True)
    return [jailbreak.get_jailbreak(behavior) for behavior in behaviors]
