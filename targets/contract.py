"""HarnessContract: the structured trace produced by a target for one conversational turn.

Scorers and reporting consume this contract instead of raw target output so that
checks (tool-call auditing, retrieval provenance, prompt-boundary leakage, etc.)
can run against a normalized shape regardless of which app the target adapts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]


@dataclass
class GeneratedImage:
    """One image artifact produced by an image-generation turn.

    Attributes:
        path: Local path to the saved image, under a gitignored output
            directory (e.g. outputs/generated/) — never committed.
        mime_type: e.g. "image/png" or "image/jpeg".
        width: Pixel width, if known.
        height: Pixel height, if known.
    """

    path: str
    mime_type: str
    width: int | None = None
    height: int | None = None


@dataclass
class HarnessContract:
    """Structured trace of a single turn against the target under test.

    Attributes:
        final_output: The user-visible text the target produced for this turn.
            May be empty for an image-only turn.
        tool_calls: Tool/function invocations the target made while producing
            final_output, in call order.
        retrieved_chunk_ids: Identifiers of any retrieval-augmented-generation
            chunks consulted while producing final_output.
        system_prompt_boundary: Index into final_output (or the target's raw
            trace, per-adapter) marking where system-prompt-controlled content
            ends and model/user-influenced content begins.
        generated_images: Image artifacts produced by this turn, if any.
        outcome: "generated" (the model produced content — text and/or
            images), "blocked" (the generation API's own safety filter
            refused the request — this is a PASS for a safeguard test, not a
            harness failure), or "error" (the harness itself failed to
            complete the turn, e.g. a timeout or transport error; see
            targets.prompt_sending_adapter.BLOCKED_PREFIX /
            targets.prompt_sending_adapter.is_blocked for the text-target
            convention this mirrors).
        request_echo: The exact prompt submitted, plus a converter-chain
            identifier if the prompt went through PyRIT converters, for
            auditability.
    """

    final_output: str
    tool_calls: list[ToolCall]
    retrieved_chunk_ids: list[str]
    system_prompt_boundary: int
    generated_images: list[GeneratedImage] = field(default_factory=list)
    outcome: Literal["generated", "blocked", "error"] = "generated"
    request_echo: str = ""
