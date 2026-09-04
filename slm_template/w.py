from typing import Iterable

import numpy as np

from .types import Vector


class WeightFunction:
    """
    First-class functional wrapper for weight parameters.

    The NumPy parameter array is owned internally. Callers may read it through
    the read-only `array` property, or call the object to obtain the current
    weight vector for vectorized NumPy operations.
    """

    def __init__(self, values: Iterable[float]):
        self._array = np.array(values, copy=True)

    @property
    def array(self) -> Vector:
        """Read-only NumPy view of the internal weight array."""
        view = self._array.view()
        view.flags.writeable = False
        return view

    def __call__(self) -> Vector:
        """Return the current weight vector as a read-only NumPy view."""
        return self.array
