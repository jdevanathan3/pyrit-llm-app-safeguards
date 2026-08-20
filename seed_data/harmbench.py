"""Shared loader for PyRIT's built-in HarmBench seed dataset.

HarmBench (https://github.com/centerforaisafety/HarmBench) is a standardized
set of ~400 harmful behaviors across categories including CBRN, cybercrime,
harassment, and misinformation. Fetched once per process and cached by PyRIT.
"""

from __future__ import annotations

from pyrit.datasets.seed_datasets.remote.harmbench_dataset import _HarmBenchDataset


async def load_harmbench_behaviors(*, category: str | None = None, limit: int = 3) -> list[str]:
    """Fetch HarmBench behaviors, optionally filtered by harm category.

    Args:
        category: If given, only behaviors tagged with this harm category are
            returned (e.g. "CBRN"). See HarmBench's own taxonomy for valid values.
        limit: Maximum number of behaviors to return.

    Returns:
        list[str]: Behavior strings (the raw harmful ask).
    """
    dataset = await _HarmBenchDataset().fetch_dataset_async()
    seeds = dataset.seeds
    if category is not None:
        seeds = [s for s in seeds if s.harm_categories and category in s.harm_categories]
    return [s.value for s in seeds[:limit]]
