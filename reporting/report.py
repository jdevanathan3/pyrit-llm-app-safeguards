"""TestResult: per-test, per-model pass/fail outcomes, and markdown rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CaseOutcome:
    model: str
    passed: bool
    prompt_excerpt: str
    response_excerpt: str
    detail: str


@dataclass
class TestResult:
    name: str
    description: str
    outcomes: list[CaseOutcome] = field(default_factory=list)

    def pass_rate(self, *, model: str) -> tuple[int, int]:
        cases = [o for o in self.outcomes if o.model == model]
        return sum(1 for o in cases if o.passed), len(cases)

    def models(self) -> list[str]:
        seen: list[str] = []
        for o in self.outcomes:
            if o.model not in seen:
                seen.append(o.model)
        return seen


def render_markdown_summary(results: list[TestResult]) -> str:
    """Render a per-test, per-model pass-rate table as markdown."""
    all_models: list[str] = []
    for r in results:
        for m in r.models():
            if m not in all_models:
                all_models.append(m)

    header = "| Test | " + " | ".join(all_models) + " |"
    sep = "|---" * (len(all_models) + 1) + "|"
    lines = [header, sep]
    for r in results:
        cells = []
        for m in all_models:
            passed, total = r.pass_rate(model=m)
            cells.append(f"{passed}/{total}")
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
