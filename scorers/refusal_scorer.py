"""Refusal scorer factory, shared by Tests 1, 2, and 3.

Wraps PyRIT's SelfAskRefusalScorer, which uses a judge LLM to determine
whether a response refuses the given objective. True = refused.
"""

from __future__ import annotations

from pyrit.prompt_target import PromptTarget
from pyrit.score import SelfAskRefusalScorer


def build_refusal_scorer(*, judge_target: PromptTarget) -> SelfAskRefusalScorer:
    """Build a refusal scorer using a fixed judge target.

    Using the same judge for every model under comparison (rather than each
    model grading itself) keeps scoring consistent across the comparison.

    Args:
        judge_target: The PromptTarget to use as the judge LLM.

    Returns:
        SelfAskRefusalScorer: True if the response is a refusal.
    """
    return SelfAskRefusalScorer(chat_target=judge_target)
