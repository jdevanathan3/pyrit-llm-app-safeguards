"""Seed prompts for Test 2: harmful-content baseline (no jailbreak wrapper).

Direct, unwrapped HarmBench behaviors — Anthropic's Unified Harm Framework
(physical, psychological, economic, societal, individual-autonomy harm):
https://www.anthropic.com/news/building-safeguards-for-claude
"""

from __future__ import annotations

from seed_data.harmbench import load_harmbench_behaviors


async def load_harm_baseline_prompts(*, limit: int = 3) -> list[str]:
    """Fetch a small, category-diverse sample of direct harmful asks.

    Args:
        limit: Number of behaviors to return.

    Returns:
        list[str]: Unwrapped HarmBench behavior strings.
    """
    return await load_harmbench_behaviors(limit=limit)
