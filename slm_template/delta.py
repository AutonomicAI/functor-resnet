from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .types import Function, Vector


@dataclass(frozen=True)
class ScopeAll:
    """Scope predicate that admits all inputs. Pickle-friendly replacement for `lambda x: True`."""

    def __call__(self, x: Vector) -> bool:  # noqa: ARG002
        return True


@dataclass(frozen=True)
class HardGate:
    """Gate that fires only on (approximately) the anchor input."""

    anchor: Vector
    rtol: float = 1e-9
    atol: float = 0.0

    def __call__(self, x: Vector) -> Vector:
        return (
            np.array([1.0])
            if np.allclose(x, self.anchor, rtol=self.rtol, atol=self.atol)
            else np.array([0.0])
        )


@dataclass(frozen=True)
class ConstantDelta:
    """Delta function that returns a constant vector (computed at anchor)."""

    df0: Vector

    def __call__(self, x: Vector) -> Vector:  # noqa: ARG002
        return self.df0


@dataclass
class TrainingEvent:
    """Structured training event with metadata."""

    x: Vector
    y: Vector
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class PatchMetadata:
    """Audit trail for each patch."""

    timestamp: float
    event: TrainingEvent
    version: int
    validation_samples: List[Vector]
    validation_passed: bool

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "event": self.event.to_dict(),
            "version": self.version,
            "validation_samples": [s.tolist() for s in self.validation_samples],
            "validation_passed": self.validation_passed,
        }


@dataclass
class DeltaPatch:
    """Additive functional delta: f'(x) = f(x) + Π(x)·Δf(x)."""

    gate: Function  # Π(x) in [0,1]
    delta: Function  # Δf(x)
    metadata: Optional[PatchMetadata] = None
    anchor: Optional[Vector] = None
    target: Optional[Vector] = None
    gate_type: str = "hard"

    def validate_gate_bounds(self, x: Vector) -> bool:
        """Verify gate returns values in [0, 1]."""
        g = float(np.max(self.gate(x)))
        return 0.0 <= g <= 1.0
