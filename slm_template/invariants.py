from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import numpy as np

from .delta import DeltaPatch
from .types import Function, Vector


@dataclass
class Invariants:
    """Container for invariants to validate deltas."""

    max_delta: float  # epsilon bound on changes
    scope_fn: Callable[[Vector], bool]  # returns True if x is in scope S
    check_monotonicity: bool = False
    check_calibration: bool = False
    custom_validators: List[Callable] = field(default_factory=list)

    def check(self, f: Function, delta: DeltaPatch, samples: Iterable[Vector]) -> bool:
        """Validate a proposed patch without constructing a composed function."""
        for x in samples:
            g = np.asarray(delta.gate(x), dtype=float)

            # Check gate bounds
            if np.any(g < 0.0) or np.any(g > 1.0):
                return False

            # Check scope constraint
            if np.any(g > 0.0) and not self.scope_fn(x):
                return False

            # Preview proposed change without constructing a composed function.
            fx = f(x)
            df = delta.delta(x)

            if g.shape == df.shape:
                fpx = fx + g * df
            else:
                fpx = fx + float(np.max(g)) * df

            # Check magnitude bound
            if np.max(np.abs(fpx - fx)) > self.max_delta:
                return False

            for validator in self.custom_validators:
                if not validator(f, delta, x):
                    return False

        return True


@dataclass
class ModelConfig:
    """Configuration for MicroModel behavior."""

    enable_skip_training: bool = True
    enable_delay_training: bool = True
    enable_audit_logging: bool = True
    enable_profiling: bool = False
    max_patches: Optional[int] = None  # Limit number of patches
