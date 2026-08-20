"""TargetAdapter: the boundary between PyRIT attack strategies and the app under test.

Each app being safety-tested gets one adapter implementation that knows how to
drive that app for a single turn and normalize its response into a HarnessContract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from targets.contract import HarnessContract


class TargetAdapter(ABC):
    """Runs a single prompt against an app under test and returns its trace."""

    @abstractmethod
    async def run_turn(self, prompt: str) -> HarnessContract:
        """Send prompt to the target application for one turn.

        Args:
            prompt: The prompt to send for this turn.

        Returns:
            HarnessContract: The structured trace of the target's response.
        """
        raise NotImplementedError
