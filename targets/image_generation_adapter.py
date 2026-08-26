"""ImageGenerationAdapter: drives a text-to-image model and normalizes its output.

Provider-agnostic: the adapter takes an injected ImageClient, so additional
providers can be added without touching the adapter itself. GeminiImageClient
is the first implementation, wrapping the Gemini image API. See
MULTIMODAL_API_NOTES.md for the verified request/response shape — in
particular, the "blocked" shape is *not* what you'd assume (no SAFETY finish
reason, no empty candidates list): a blocked request comes back with
finish_reason STOP and a text-only refusal Part instead of image bytes.

Outputs are saved under a gitignored directory and never committed — this
harness routinely generates images of synthetic/test subjects and, when
pointed at operator-supplied descriptive prompts (seed_data/descriptive_likeness_prompts.py),
may generate a recognizable likeness of a real person as the very failure
mode under test. See README's "Multimodal likeness testing" section for the
retention/deletion policy.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from targets.adapter import TargetAdapter
from targets.contract import GeneratedImage, HarnessContract
from targets.prompt_sending_adapter import BLOCKED_PREFIX, call_with_timeout

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/generated")

# Model ids from the implementation brief (the "Nano Banana" family). Passed
# explicitly to GeminiImageClient rather than hardcoded in the adapter, and
# logged per-request for auditability. "-preview" ids can be renamed
# upstream without notice (this project already hit that with a text model:
# gemini-3.1-pro-preview -> gemini-3.1-pro), so treat these as current
# defaults to verify against MULTIMODAL_API_NOTES.md, not guarantees.
NANO_BANANA_PRO = "gemini-3-pro-image-preview"
NANO_BANANA_2 = "gemini-3.1-flash-image-preview"
NANO_BANANA = "gemini-2.5-flash-image"

# Image-generation-specific and generic blocking finish reasons documented by
# the google-genai SDK. Only IMAGE_SAFETY-shaped responses were reproduced
# live (see MULTIMODAL_API_NOTES.md); the rest are handled defensively.
_BLOCKING_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "RECITATION",
        "BLOCKLIST",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "NO_IMAGE",
    }
)


@dataclass
class RawImage:
    data: bytes
    mime_type: str


@dataclass
class ImageGenerationOutcome:
    """Normalized result from an ImageClient.generate_async call."""

    outcome: Literal["generated", "blocked", "error"]
    images: list[RawImage]
    text: str
    detail: str


class ImageClient(Protocol):
    """Provider-agnostic interface an ImageGenerationAdapter drives."""

    model_id: str

    async def generate_async(self, prompt: str) -> ImageGenerationOutcome: ...


class GeminiImageClient:
    """Wraps the Gemini image-generation API (google-genai SDK)."""

    def __init__(self, *, api_key: str, model_id: str = NANO_BANANA_PRO) -> None:
        from google import genai  # local import: keep this an optional dependency

        self.model_id = model_id
        self._client = genai.Client(api_key=api_key)

    async def generate_async(self, prompt: str) -> ImageGenerationOutcome:
        logger.info("Requesting image from model=%s", self.model_id)
        response = await self._client.aio.models.generate_content(model=self.model_id, contents=prompt)

        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
        if block_reason:
            return ImageGenerationOutcome(
                outcome="blocked", images=[], text="", detail=f"prompt_feedback.block_reason={block_reason}"
            )

        if not response.candidates:
            return ImageGenerationOutcome(outcome="blocked", images=[], text="", detail="empty candidates list")

        candidate = response.candidates[0]
        finish_reason = str(getattr(candidate, "finish_reason", ""))
        finish_reason_name = finish_reason.rsplit(".", 1)[-1]  # "FinishReason.STOP" -> "STOP"

        images: list[RawImage] = []
        text_parts: list[str] = []
        parts = (candidate.content.parts if candidate.content else None) or []
        for part in parts:
            if getattr(part, "inline_data", None) is not None:
                images.append(RawImage(data=part.inline_data.data, mime_type=part.inline_data.mime_type))
            elif getattr(part, "text", None):
                text_parts.append(part.text)
        text = "\n".join(text_parts)

        if finish_reason_name in _BLOCKING_FINISH_REASONS:
            return ImageGenerationOutcome(
                outcome="blocked", images=[], text=text, detail=f"finish_reason={finish_reason_name}"
            )

        # The confirmed-live blocked shape: STOP with a text refusal and no image data.
        if not images:
            return ImageGenerationOutcome(
                outcome="blocked",
                images=[],
                text=text,
                detail=f"finish_reason={finish_reason_name} but no inline_data part returned",
            )

        return ImageGenerationOutcome(outcome="generated", images=images, text=text, detail=f"finish_reason={finish_reason_name}")


class ImageGenerationAdapter(TargetAdapter):
    """Drives a text-to-image ImageClient and normalizes its output into a HarnessContract."""

    def __init__(self, *, client: ImageClient, name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        """
        Args:
            client: The provider-specific image client to drive (e.g. GeminiImageClient).
            name: Identifier for this adapter (for comparison/reporting).
            output_dir: Gitignored directory generated images are saved under.
        """
        self.name = name
        self._client = client
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run_turn(self, prompt: str) -> HarnessContract:
        """
        Returns:
            HarnessContract with generated_images populated on success,
            outcome="blocked" if the provider's own safety filter refused
            (a pass for the safeguard under test), or outcome="error" with
            final_output prefixed BLOCKED_PREFIX if the harness itself
            failed to complete the turn (timeout, transport error, etc.) —
            mirroring targets.prompt_sending_adapter's convention.
        """
        try:
            result = await call_with_timeout(self._client.generate_async(prompt))
        except Exception as e:
            logger.warning("Image client %s raised for prompt, treating as error: %s", self.name, e)
            return HarnessContract(
                final_output=f"{BLOCKED_PREFIX} image client raised {type(e).__name__}: {e}]",
                tool_calls=[],
                retrieved_chunk_ids=[],
                system_prompt_boundary=0,
                outcome="error",
                request_echo=prompt,
            )

        if result.outcome == "blocked":
            return HarnessContract(
                final_output=result.text,
                tool_calls=[],
                retrieved_chunk_ids=[],
                system_prompt_boundary=0,
                outcome="blocked",
                request_echo=prompt,
            )

        saved = [self._save_image(image) for image in result.images]
        return HarnessContract(
            final_output=result.text,
            tool_calls=[],
            retrieved_chunk_ids=[],
            system_prompt_boundary=0,
            generated_images=saved,
            outcome="generated",
            request_echo=prompt,
        )

    def _save_image(self, image: RawImage) -> GeneratedImage:
        extension = image.mime_type.rsplit("/", 1)[-1] or "bin"
        filename = f"{uuid.uuid4()}.{extension}"
        path = self._output_dir / filename
        path.write_bytes(image.data)
        return GeneratedImage(path=str(path), mime_type=image.mime_type)
