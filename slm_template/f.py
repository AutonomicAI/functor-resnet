from dataclasses import dataclass, field
from typing import Dict, List
import hashlib

import numpy as np

from .delta import DeltaPatch
from .types import Function, Vector
from .w import WeightFunction


@dataclass
class MaterializedLearnedFunction:
    """
    Materialized learned function: f_new = f_old + delta, as a direct sum.

    Each one-shot commit is an exact override at its anchor, stored as a lookup
    entry (f'(anchor) = target = f(anchor) + Δf(anchor)). This avoids nested
    closure chains while preserving f(x) as the inference API.
    """

    base_f: Function
    trained_outputs: Dict[str, Vector] = field(default_factory=dict)

    @staticmethod
    def hash_input(x: Vector) -> str:
        """
        Return a canonical hash for lookup keys.

        Canonicalization prevents silent hard-gate misses across dtype/layout variants
        such as np.array([2]), np.array([2.0]), float32 vs float64, or non-contiguous
        views. Values are encoded as little-endian float64 in C order.
        """
        arr = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
        shape = repr(arr.shape).encode("utf-8")
        return hashlib.sha256(shape + b"|" + arr.tobytes()).hexdigest()

    def __call__(self, x: Vector) -> Vector:
        x_hash = self.hash_input(x)
        if x_hash in self.trained_outputs:
            return self.trained_outputs[x_hash]
        return self.base_f(x)

    def clone(self) -> "MaterializedLearnedFunction":
        """
        Return a shallow structural clone for shadow preparation.

        The base function is shared; the mutable lookup table is copied so a new
        patch can be learned without mutating the currently published function.
        """
        return MaterializedLearnedFunction(
            base_f=self.base_f,
            trained_outputs={k: v.copy() for k, v in self.trained_outputs.items()},
        )

    def learn(self, patch: DeltaPatch) -> None:
        if patch.anchor is None or patch.target is None:
            raise ValueError("One-shot commits require anchor and target metadata.")
        self.trained_outputs[self.hash_input(patch.anchor)] = patch.target.copy()

    def reset_to(self, patches: List[DeltaPatch]) -> None:
        self.trained_outputs.clear()
        for patch in patches:
            self.learn(patch)


def base_model_sigmoid(x: Vector) -> Vector:
    """Example: independent probabilities (sigmoid outputs)."""
    logits = np.array([np.sin(x[0]), np.cos(x[0])])
    return 1 / (1 + np.exp(-logits))


LINEAR_BASE_WEIGHTS = WeightFunction([1.0, 0.5])


def base_model_linear(x: Vector) -> Vector:
    """Example: simple linear model."""
    # TODO: Replace with actual learned weights
    return LINEAR_BASE_WEIGHTS() * x[0]


def base_model_lookup(lookup_table: Dict[str, Vector]) -> Function:
    """Example: lookup table base function."""

    def lookup(x: Vector) -> Vector:
        key = str(x.tolist())
        return lookup_table.get(key, np.zeros_like(x))

    return lookup
