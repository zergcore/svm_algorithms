from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class Kernels:
    """Implementaciones vectorizadas de funciones Kernel."""

    @staticmethod
    def linear() -> callable:
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return X1 @ X2.T
        return calc

    @staticmethod
    def gaussian(gamma: float = 0.5) -> callable:
        """Kernel Gaussiano (Radial Basis Function)."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            sq_norm1 = np.sum(X1**2, axis=1, keepdims=True)
            sq_norm2 = np.sum(X2**2, axis=1)
            dist_sq = sq_norm1 + sq_norm2 - 2 * (X1 @ X2.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            return np.exp(-gamma * dist_sq)
        return calc

    @staticmethod
    def polynomial(degree: float = 3, c: float = 1.0) -> callable:
        """Kernel Polinomial."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return (X1 @ X2.T + c) ** degree
        return calc

    @staticmethod
    def sigmoid(gamma: float = 1.0, r: float = 0.0) -> callable:
        """Kernel Sigmoide."""
        def calc(X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.tanh(gamma * (X1 @ X2.T) + r)
        return calc
