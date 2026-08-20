"""PromptSendingAdapter: drives any PyRIT PromptTarget through PromptSendingAttack.

One TargetAdapter implementation reused across every model under comparison —
only the wrapped PromptTarget differs between an OpenAI run and an Anthropic run.
"""

from __future__ import annotations

import logging

from pyrit.executor.attack import PromptSendingAttack
from pyrit.models import Message
from pyrit.prompt_target import PromptTarget

from targets.adapter import TargetAdapter
from targets.contract import HarnessContract

logger = logging.getLogger(__name__)

BLOCKED_PREFIX = "[BLOCKED:"


def is_blocked(contract: HarnessContract) -> bool:
    """True if the target raised while handling this turn (see run_turn)."""
    return contract.final_output.startswith(BLOCKED_PREFIX)


class PromptSendingAdapter(TargetAdapter):
    """Sends a prompt to a PyRIT PromptTarget via PromptSendingAttack and normalizes the result."""

    def __init__(self, *, target: PromptTarget, name: str) -> None:
        """
        Args:
            target: The PyRIT PromptTarget to drive (e.g. an OpenAIChatTarget
                pointed at OpenAI or at Anthropic's OpenAI-compatible endpoint).
            name: Identifier for this adapter, used as the key in ModelComparison.contracts.
        """
        self.name = name
        self._attack = PromptSendingAttack(objective_target=target)

    async def run_turn(self, prompt: str, *, prepended_conversation: list[Message] | None = None) -> HarnessContract:
        """
        Args:
            prompt: The prompt to send as the final turn.
            prepended_conversation: Optional conversation history to seed before
                sending prompt (e.g. a system-role message carrying a system prompt).

        Returns:
            HarnessContract: Normalized result. tool_calls is always empty —
                this harness doesn't register real tools with the target yet.
                If the target raises (e.g. some providers return a hard HTTP
                error like 403 for a blocked request instead of a normal
                chat-completion refusal or content-filter response),
                final_output records the block rather than propagating the
                exception — a hard API-level rejection is itself evidence the
                safeguard held, not a harness failure.
        """
        try:
            result = await self._attack.execute_async(
                objective=prompt,
                prepended_conversation=prepended_conversation,
            )
            final_output = result.last_response.converted_value if result.last_response else ""
        except Exception as e:
            logger.warning("Target %s raised while sending prompt, treating as blocked: %s", self.name, e)
            final_output = f"{BLOCKED_PREFIX} target raised {type(e).__name__}: {e}]"

        return HarnessContract(
            final_output=final_output,
            tool_calls=[],
            retrieved_chunk_ids=[],
            system_prompt_boundary=0,
        )
