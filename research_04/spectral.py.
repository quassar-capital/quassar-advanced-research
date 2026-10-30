
"""Finite-dimensional diagnostics for non-normal spectral problems.

All norms and adjoints in this module are Euclidean. A physical energy norm
requires an explicitly defined metric and must not be inferred from these
matrix diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SimpleEigenvalueSensitivity:
    eigenvalue: complex
    condition_number: float
    directional_derivative: complex
    right_residual: float
    left_residual: float


def _square_matrix(values: ArrayLike, name: str) -> NDArray[np.complex128]:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite entries")
    return matrix


def smallest_singular_value(operator: ArrayLike, z: complex) -> float:
    """Return sigma_min(z I - A), an inverse-resolvent diagnostic."""
    a = _square_matrix(operator, "operator")
    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        raise ValueError("z must be finite")
    return float(np.linalg.svd(z * np.eye(a.shape[0]) - a, compute_uv=False)[-1])


def eigenvalue_sensitivity(
    operator: ArrayLike,
    perturbation: ArrayLike,
    eigenvalue_index: int,
    *,
    separation_tolerance: float = 1e-8,
) -> SimpleEigenvalueSensitivity:
    """Analyze one numerically isolated simple eigenvalue of a finite matrix.

    Eigenvalues are ordered lexicographically by real and imaginary parts.
    This ordering is deterministic, not a mode-tracking rule. The tolerance
    measures relative spectral separation and is not a proof of simplicity.
    """
    a = _square_matrix(operator, "operator")
    e = _square_matrix(perturbation, "perturbation")
    if e.shape != a.shape:
        raise ValueError("operator and perturbation must have equal dimensions")
    if not np.isfinite(separation_tolerance) or separation_tolerance <= 0:
        raise ValueError("separation_tolerance must be positive and finite")
    eigenvalues, eigenvectors = np.linalg.eig(a)
    order = np.lexsort((eigenvalues.imag, eigenvalues.real))
    if not isinstance(eigenvalue_index, (int, np.integer)) or not 0 <= eigenvalue_index < len(order):
        raise IndexError("eigenvalue_index is outside the spectrum")
    chosen = order[eigenvalue_index]
    lam = complex(eigenvalues[chosen])
    scale = max(1.0, float(np.linalg.norm(a, ord=2)), abs(lam))
    if len(order) > 1:
        gap = min(abs(lam - eigenvalues[j]) for j in range(len(order)) if j != chosen)
        if gap <= separation_tolerance * scale:
            raise ValueError("eigenvalue is not sufficiently isolated")
    v = eigenvectors[:, chosen]
    # The smallest right singular vector of A* - conjugate(lambda) I
    # is a left eigenvector approximation associated with lambda.
    adjoint_shift = a.conj().T - lam.conjugate() * np.eye(a.shape[0])
    _, _, vh = np.linalg.svd(adjoint_shift)
    w = vh[-1, :].conj()
    overlap = np.vdot(w, v)
    if abs(overlap) <= 100 * np.finfo(float).eps * np.linalg.norm(w) * np.linalg.norm(v):
        raise ValueError("left-right eigenvector overlap is numerically singular")
    condition = float(np.linalg.norm(w) * np.linalg.norm(v) / abs(overlap))
    derivative = complex(np.vdot(w, e @ v) / overlap)
    right_residual = float(np.linalg.norm(a @ v - lam * v))
    left_residual = float(np.linalg.norm(a.conj().T @ w - lam.conjugate() * w))
    return SimpleEigenvalueSensitivity(lam, condition, derivative, right_residual, left_residual)
