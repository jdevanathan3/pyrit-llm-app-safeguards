"""ModelComparison: the same prompt's HarnessContract from each model under comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

from targets.adapter import TargetAdapter
from targets.contract import HarnessContract


@dataclass
class ModelComparison:
    """Structured result of running one prompt through multiple TargetAdapters.

    Attributes:
        prompt: The prompt sent to every adapter.
        contracts: Each adapter's HarnessContract, keyed by adapter name.
    """

    prompt: str
    contracts: dict[str, HarnessContract] = field(default_factory=dict)


async def run_comparison_async(*, prompt: str, adapters: dict[str, TargetAdapter]) -> ModelComparison:
    """Run the same prompt through every adapter and collect each HarnessContract.

    Args:
        prompt: The prompt to send to every adapter.
        adapters: TargetAdapter instances to run, keyed by name.

    Returns:
        ModelComparison: One HarnessContract per adapter, keyed the same way.
    """
    comparison = ModelComparison(prompt=prompt)
    for name, adapter in adapters.items():
        comparison.contracts[name] = await adapter.run_turn(prompt)
    return comparison
