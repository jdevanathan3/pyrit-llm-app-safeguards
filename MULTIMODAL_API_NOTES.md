# Multimodal API notes (verified against installed packages, 2026-08-26)

Per the implementation brief's Section 0.2: these are findings from actually
calling the installed SDKs, not assumptions from memory. Two things below
turned out to differ from what memory would have suggested.

## Gemini image generation (`google-genai`)

- Package: `google-genai` (`from google import genai`), not previously
  installed — added to `pyproject.toml`.
- Client: `genai.Client(api_key=...)`.
- Call: `client.models.generate_content(model=<id>, contents=<prompt str>)`
  — synchronous, returns pixels directly in the response (confirmed no job
  ID / polling for this model family, per brief Section 0.3).
- **Success shape:** `response.candidates[0].finish_reason ==
  FinishReason.STOP`, and `response.candidates[0].content.parts[0]` has
  `.inline_data.mime_type` (e.g. `"image/jpeg"`) and `.inline_data.data`
  (raw bytes, not base64 text).
- **Blocked shape — verified live, and it is NOT what you'd assume:** a
  policy-violating prompt (tested with a generic violence/gore prompt, no
  named individual) still comes back with `finish_reason == STOP` and a
  *populated* `candidates` list — but the returned `Part` has `.text` set
  ("I can't generate images that are sexually explicit, gory, or violent.")
  and `.inline_data` is `None`. There is no `SAFETY` finish reason, no
  empty-candidates response, and `response.prompt_feedback` is `None` in
  this case. **The only reliable signal that a request was blocked is the
  absence of an `inline_data` part — `finish_reason` alone is not enough.**
- **A second, harder blocked shape was also observed** for a more severely
  worded prompt (still generic/violence-category, no named individual): the
  response has `finish_reason == STOP` but `candidate.content.parts` is
  `None` (not an empty list) — no text, no image, nothing. A naive
  `for part in candidate.content.parts` crashes with `TypeError: 'NoneType'
  object is not iterable` on this shape; `image_generation_adapter.py`
  guards for `parts is None` explicitly because of this. Also worth noting:
  the *same* violence-category prompt did not reliably block on every call —
  safety classification for this category was observed to be non-deterministic
  across repeated identical requests, same as this repo's LLM-judge scorers.
- The SDK also defines dedicated image-specific finish reasons —
  `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`,
  `IMAGE_OTHER`, `NO_IMAGE` — plus the generic `SAFETY`, `PROHIBITED_CONTENT`,
  `RECITATION`, `BLOCKLIST`, `SPII`. These were not observed in testing
  (only the STOP-with-text-only-refusal shape was), but `image_generation_adapter.py`
  treats any of them as blocked too, defensively, since the SDK reference
  documents them as blocking outcomes for this model family.
- `response.prompt_feedback.block_reason` (pre-generation prompt block) is
  also checked defensively even though it didn't fire in either test above.
- Model ids tried live: `gemini-3-pro-image-preview` (Nano Banana Pro) —
  works as of this writing. Per the brief, `gemini-3.1-flash-image-preview`
  and `gemini-2.5-flash-image` are the cheaper dev-lane ids; not
  individually re-verified live (same client/response shape expected since
  they're the same API family) — the adapter takes the model id as a
  constructor argument so this is a config change, not a code change, if a
  ".../-preview" id gets renamed the way `gemini-3.1-pro-preview` did
  earlier in this project's chat-model work.

## Face detection + embedding (`insightface` + `onnxruntime`)

- Not previously installed. Neither PyRIT nor its dependencies pull in a
  face-recognition stack, so this is new: `insightface`, `onnxruntime`,
  `opencv-python-headless` added to `pyproject.toml`. All installed with
  prebuilt wheels for the project's Python (3.14, macOS arm64) — no native
  compilation needed (this ruled out `face_recognition`/`dlib`, which
  generally requires a from-source C++ build).
- API: `insightface.app.FaceAnalysis(name="buffalo_l")`, `.prepare(ctx_id=...)`,
  `.get(bgr_image_ndarray)` → list of `Face` objects, each with `.embedding`
  (512-d ArcFace vector) and `.bbox`. First use downloads the `buffalo_l`
  model bundle (~350MB) from insightface's model zoo — this needs network
  access once, then it's cached under `~/.insightface`. This is a real
  runtime dependency worth flagging: the likeness scorer will fail on first
  use in a fully offline environment until that download completes.
- `Face` objects also carry the model's own `.age`/`.sex` estimates. These
  are **not** used as demographic ground truth anywhere in this
  implementation — bias validation (Section 4 of the brief) requires
  externally-labeled demographic slices from the operator's validation set,
  not the detector's own guesses about the images it's being validated
  against (that would be circular).
- `sklearn` is not installed and was not added — threshold calibration
  (ROC-style sweep) is implemented directly against cosine similarity with
  plain numpy in `scorers/likeness_calibration.py`, since it's a small,
  well-understood computation that didn't justify a new heavy dependency.

## PyRIT 1.0.1

No new PyRIT-specific API surface was needed beyond what's already in use
elsewhere in this repo (`TargetAdapter`, `HarnessContract`,
`call_with_timeout`/`is_blocked` conventions, the `build_*_scorer` factory
pattern). The likeness scorer is deliberately **not** a PyRIT `Scorer`
subclass — it operates on `HarnessContract.generated_images` (local file
paths) rather than a `Message`/text response, which doesn't fit PyRIT's
`Scorer.score_text_async`/`score_async` shape. It follows the same
`build_*_scorer(...)` factory naming convention as `scorers/refusal_scorer.py`
for consistency, but returns a plain callable/class rather than a PyRIT
`Scorer`.
