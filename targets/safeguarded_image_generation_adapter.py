"""SafeguardedImageGenerationAdapter: wraps image generation with both
prompt-level and image-level real-person/minor safeguards.

Composes:
1. scorers/prompt_safety_screen.py (Safeguard 1) — runs BEFORE generation.
   A flagged prompt never reaches the image generator at all.
2. The underlying ImageGenerationAdapter — only reached if (1) passes.
3. scorers/image_safety_screen.py (Safeguard 2) — runs on every generated
   image AFTER generation, as a backstop for prompts that evaded (1).

Either safeguard firing produces outcome="blocked" with a user-facing
prevention message in final_output, same convention as a provider-side
safety block (targets/image_generation_adapter.py's ImageGenerationOutcome)
— from the caller's perspective, a safeguard holding is a safeguard holding,
regardless of which layer caught it. The generated_images list is emptied
when Safeguard 2 blocks, even though the file itself is retained on disk
for audit (see README's Multimodal likeness testing retention/deletion
policy) — the point of "prevention" is that a blocked image is never
returned as if it were successfully delivered content.
"""

from __future__ import annotations

import logging

from targets.adapter import TargetAdapter
from targets.contract import HarnessContract
from targets.image_generation_adapter import ImageGenerationAdapter
from scorers.image_safety_screen import ImageSafetyScreen
from scorers.prompt_safety_screen import PromptSafetyScreen

logger = logging.getLogger(__name__)

PREVENTION_MESSAGE = "I can't generate images of real, identifiable people or of minors."


class SafeguardedImageGenerationAdapter(TargetAdapter):
    def __init__(
        self,
        *,
        underlying_adapter: ImageGenerationAdapter,
        prompt_screen: PromptSafetyScreen,
        image_screen: ImageSafetyScreen,
        name: str = "safeguarded-image-generation",
    ) -> None:
        self.name = name
        self._underlying_adapter = underlying_adapter
        self._prompt_screen = prompt_screen
        self._image_screen = image_screen

    async def run_turn(self, prompt: str) -> HarnessContract:
        prompt_result = await self._prompt_screen.screen_async(prompt)
        if prompt_result.blocked:
            logger.info("Safeguard 1 (prompt screen) blocked prompt: %s", prompt_result.reason)
            return HarnessContract(
                final_output=PREVENTION_MESSAGE,
                tool_calls=[],
                retrieved_chunk_ids=[],
                system_prompt_boundary=0,
                outcome="blocked",
                request_echo=f"{prompt} [blocked_by=prompt_screen; reason={prompt_result.reason}]",
            )

        contract = await self._underlying_adapter.run_turn(prompt)
        if contract.outcome != "generated":
            return contract  # provider itself already blocked, or the harness hit an error

        for image in contract.generated_images:
            image_result = await self._image_screen.screen_async(image.path)
            if image_result.blocked:
                logger.info(
                    "Safeguard 2 (image screen) blocked generated image %s: %s", image.path, image_result.reason
                )
                return HarnessContract(
                    final_output=PREVENTION_MESSAGE,
                    tool_calls=[],
                    retrieved_chunk_ids=[],
                    system_prompt_boundary=0,
                    outcome="blocked",
                    request_echo=f"{prompt} [blocked_by=image_screen; reason={image_result.reason}]",
                )

        return contract
