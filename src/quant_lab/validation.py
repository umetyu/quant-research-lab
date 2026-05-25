from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    """Expanding-window walk-forward splitter for ordered observations."""

    train_size: int
    test_size: int
    step_size: int | None = None

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.train_size <= 0 or self.test_size <= 0:
            raise ValueError("train_size and test_size must be positive")

        step = self.step_size or self.test_size
        start = self.train_size

        while start + self.test_size <= n_samples:
            train_idx = np.arange(0, start)
            test_idx = np.arange(start, start + self.test_size)
            yield train_idx, test_idx
            start += step
