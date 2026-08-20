"""HarnessContract: the structured trace produced by a target for one conversational turn.

Scorers and reporting consume this contract instead of raw target output so that
checks (tool-call auditing, retrieval provenance, prompt-boundary leakage, etc.)
can run against a normalized shape regardless of which app the target adapts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]


@dataclass
class HarnessContract:
    """Structured trace of a single turn against the target under test.

    Attributes:
        final_output: The user-visible text the target produced for this turn.
        tool_calls: Tool/function invocations the target made while producing
            final_output, in call order.
        retrieved_chunk_ids: Identifiers of any retrieval-augmented-generation
            chunks consulted while producing final_output.
        system_prompt_boundary: Index into final_output (or the target's raw
            trace, per-adapter) marking where system-prompt-controlled content
            ends and model/user-influenced content begins.
    """

    final_output: str
    tool_calls: list[ToolCall]
    retrieved_chunk_ids: list[str]
    system_prompt_boundary: int
