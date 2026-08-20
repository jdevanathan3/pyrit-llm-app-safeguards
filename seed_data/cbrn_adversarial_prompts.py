"""Seed prompts for Test 3: CBRN/catastrophic-uplift refusal under adversarial framing.

HarmBench behaviors tagged CBRN, wrapped in jailbreak templates. Anthropic's
Responsible Scaling Policy names CBRN uplift as the specific catastrophic-risk
domain ASL-3 deployment safeguards must hold "under adversarial testing":
https://www.anthropic.com/responsible-scaling-policy
"""

from __future__ import annotations

from pyrit.datasets import TextJailBreak

from seed_data.harmbench import load_harmbench_behaviors


async def load_cbrn_adversarial_prompts(*, limit: int = 3) -> list[str]:
    """Build jailbreak-wrapped CBRN-category harmful prompts.

    Args:
        limit: Number of CBRN-tagged HarmBench behaviors to wrap.

    Returns:
        list[str]: Jailbreak-template-wrapped CBRN prompts.
    """
    behaviors = await load_harmbench_behaviors(category="CBRN", limit=limit)
    jailbreak = TextJailBreak(random_template=True)
    return [jailbreak.get_jailbreak(behavior) for behavior in behaviors]
