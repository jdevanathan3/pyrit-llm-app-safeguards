# pyrit-llm-app-safeguards

App-level LLM safety test suite built on [PyRIT](https://github.com/Azure/PyRIT).

Tests target the *application* wrapped around an LLM (tool use, retrieval,
system-prompt boundaries) rather than the base model, by normalizing every
turn's response into a `HarnessContract` that scorers and reporting can
check consistently across different apps.

## Status

Scaffold only — the `HarnessContract` data shape and `TargetAdapter` interface
are defined; no scorers, datasets, or test logic have been implemented yet.

## Layout

- `targets/` — `HarnessContract` and `TargetAdapter`; one adapter per app under
  test, each responsible for driving that app and normalizing its response.
- `datasets/` — PyRIT `SeedDataset`/`SeedPrompt` definitions and loaders for
  attack objectives and seed prompts.
- `scorers/` — PyRIT `Scorer` implementations that evaluate a `HarnessContract`
  (tool-call auditing, retrieval provenance, prompt-boundary leakage, etc.).
- `tests/` — the test suite itself, wiring datasets + target adapters + scorers
  together via PyRIT attack strategies.
- `reporting/` — aggregation and output of scoring results.

## Requirements

- Python >= 3.10
- [PyRIT](https://pypi.org/project/pyrit/) `==1.0.1` (pinned in `pyproject.toml`, installed in `.venv`)

## Environment configuration

Copy `.env.example` to `.env` and fill in real values; `.env` is gitignored.

PyRIT does **not** auto-load a project-local `.env`. `pyrit.common.default_values`
reads secrets straight from `os.environ` (no dotenv import at all), and the only
place PyRIT touches `python-dotenv` is `pyrit.setup.initialize_pyrit_async`,
which by default loads `~/.pyrit/.env` / `~/.pyrit/.env.local` — not a repo-root
`.env`. So before constructing a target, either:

- call `await initialize_pyrit_async("InMemory", env_files=[Path(".env")])`
  (or `"SQLite"`/`"AzureSQL"`) to point it at this repo's `.env`, or
- call `dotenv.load_dotenv()` yourself before constructing any target.

## License

Apache-2.0 — see [LICENSE](./LICENSE).
