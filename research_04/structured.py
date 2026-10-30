
"""Structured first-order eigenvalue sensitivity.

Perturbation subspaces are explicitly supplied by the researcher.
This module does not infer physical admissibility from matrix structure.

The Euclidean Frobenius norm is used for perturbation amplitudes.
The reported structured sensitivity is the exact first-order maximum
over the complex span of the supplied basis, for a simple eigenvalue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class StructuredSensitivity:
    eigenvalue: complex
    unstructured_bound: float
    structured_bound: float
    relative_sensitivity: float
    perturbation: NDArray[np.complex128]
    directional_derivative: complex
    basis_rank: int


def _matrix(values: ArrayLike, name: str) -> NDArray[np.complex128]:
    a = np.asarray(values, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite entries")
    return a


def structured_eigenvalue_sensitivity(
    operator: ArrayLike,
    eigenvalue_index: int,
    perturbation_basis: list[ArrayLike],
    *,
    separation_tolerance: float = 1e-8,
    rank_tolerance: float | None = None,
) -> StructuredSensitivity:
    """Maximize first-order sensitivity over a complex matrix subspace.

    The basis is interpreted as a complex-linear span. Real-only
    perturbation constraints require a different optimization problem.
    """
    a = _matrix(operator, "operator")
    n = a.shape[0]

    if not perturbation_basis:
        raise ValueError("perturbation_basis must not be empty")

    if not np.isfinite(separation_tolerance) or separation_tolerance <= 0:
        raise ValueError("separation_tolerance must be positive and finite")

    matrices = [_matrix(b, "basis matrix") for b in perturbation_basis]
    if any(b.shape != a.shape for b in matrices):
        raise ValueError("all basis matrices must match operator dimensions")

    values, vectors = np.linalg.eig(a)
    order = np.lexsort((values.imag, values.real))

    if (
        not isinstance(eigenvalue_index, (int, np.integer))
        or not 0 <= eigenvalue_index < n
    ):
        raise IndexError("eigenvalue_index is outside the spectrum")

    index = order[eigenvalue_index]
    lam = complex(values[index])

    scale = max(1.0, float(np.linalg.norm(a, 2)), abs(lam))
    if n > 1:
        gap = min(abs(lam - values[j]) for j in range(n) if j != index)
        if gap <= separation_tolerance * scale:
            raise ValueError("eigenvalue is not sufficiently isolated")

    v = vectors[:, index]

    adjoint_shift = a.conj().T - lam.conjugate() * np.eye(n)
    _, _, vh = np.linalg.svd(adjoint_shift)
    w = vh[-1, :].conj()

    overlap = np.vdot(w, v)
    if abs(overlap) <= 100 * np.finfo(float).eps:
        raise ValueError("left-right eigenvector overlap is numerically singular")

    # Construct an orthonormal basis for the matrix subspace under
    # the Frobenius inner product.
    columns = np.column_stack([b.reshape(-1) for b in matrices])
    u, singular_values, _ = np.linalg.svd(
        columns, full_matrices=False
    )

    if rank_tolerance is None:
        tolerance = (
            max(columns.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
    else:
        if not np.isfinite(rank_tolerance) or rank_tolerance < 0:
            raise ValueError("rank_tolerance must be finite and nonnegative")
        tolerance = rank_tolerance

    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        raise ValueError("perturbation basis has zero numerical rank")

    q = u[:, :rank]

    # The linear functional is w* E v / (w* v).
    # Its Frobenius representation is w v* / conjugate(w* v).
    representer = (
        np.outer(w, v.conj()) / overlap.conjugate()
    ).reshape(-1)

    projection = q @ (q.conj().T @ representer)
    structured_bound = float(np.linalg.norm(projection))
    unstructured_bound = float(
        np.linalg.norm(w) * np.linalg.norm(v) / abs(overlap)
    )

    if structured_bound > 0:
        optimal = (projection / structured_bound).reshape(n, n)
    else:
        optimal = np.zeros((n, n), dtype=np.complex128)

    derivative = complex(np.vdot(w, optimal @ v) / overlap)

    return StructuredSensitivity(
        eigenvalue=lam,
        unstructured_bound=unstructured_bound,
        structured_bound=structured_bound,
        relative_sensitivity=structured_bound / unstructured_bound,
        perturbation=optimal,
        directional_derivative=derivative,
        basis_rank=rank,
    )
