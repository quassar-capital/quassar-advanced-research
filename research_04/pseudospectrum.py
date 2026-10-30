
"""Pseudospectral resolvent diagnostics for Research 04.

Computes finite-dimensional Euclidean pseudospectral quantities
using singular-value decompositions.

These diagnostics characterize discretized operators. They do not,
without additional convergence analysis and an appropriate physical
norm, establish properties of the continuous Kerr operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ResolventDiagnostic:
    spectral_parameter: complex
    smallest_singular_value: float
    resolvent_norm: float
    spectral_distance: float
    amplification_ratio: float


@dataclass(frozen=True)
class PseudospectralGrid:
    real_axis: NDArray[np.float64]
    imaginary_axis: NDArray[np.float64]
    smallest_singular_values: NDArray[np.float64]
    spectral_distances: NDArray[np.float64]
    amplification_ratios: NDArray[np.float64]


def _validate_operator(
    operator: ArrayLike,
) -> NDArray[np.complex128]:
    a = np.asarray(operator, dtype=np.complex128)

    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("operator must be a nonempty square matrix")

    if not np.all(np.isfinite(a)):
        raise ValueError("operator must contain finite entries")

    return a


def _validate_axis(
    values: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    axis = np.asarray(values, dtype=np.float64)

    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")

    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values")

    if axis.size > 1 and not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing")

    return axis


def _diagnostic(
    operator: NDArray[np.complex128],
    eigenvalues: NDArray[np.complex128],
    z: complex,
) -> ResolventDiagnostic:
    identity = np.eye(operator.shape[0], dtype=np.complex128)
    shifted = z * identity - operator

    singular_values = np.linalg.svd(
        shifted,
        compute_uv=False,
    )

    sigma_min = float(singular_values[-1])
    distance = float(np.min(np.abs(z - eigenvalues)))

    resolvent_norm = (
        float("inf") if sigma_min == 0.0 else 1.0 / sigma_min
    )

    if distance == 0.0:
        amplification = float("nan")
    elif sigma_min == 0.0:
        amplification = float("inf")
    else:
        amplification = distance / sigma_min

    return ResolventDiagnostic(
        spectral_parameter=complex(z),
        smallest_singular_value=sigma_min,
        resolvent_norm=resolvent_norm,
        spectral_distance=distance,
        amplification_ratio=amplification,
    )


def resolvent_diagnostic(
    operator: ArrayLike,
    spectral_parameter: complex,
) -> ResolventDiagnostic:
    """Evaluate resolvent sensitivity at one complex parameter."""
    a = _validate_operator(operator)
    z = complex(spectral_parameter)

    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        raise ValueError("spectral_parameter must be finite")

    eigenvalues = np.linalg.eigvals(a)

    return _diagnostic(a, eigenvalues, z)


def pseudospectral_grid(
    operator: ArrayLike,
    real_axis: ArrayLike,
    imaginary_axis: ArrayLike,
) -> PseudospectralGrid:
    """Evaluate singular-value diagnostics on a complex spectral grid.

    Output arrays have shape (len(imaginary_axis), len(real_axis)).
    """
    a = _validate_operator(operator)
    x = _validate_axis(real_axis, "real_axis")
    y = _validate_axis(imaginary_axis, "imaginary_axis")

    eigenvalues = np.linalg.eigvals(a)

    shape = (y.size, x.size)

    sigma = np.empty(shape, dtype=np.float64)
    distances = np.empty(shape, dtype=np.float64)
    amplification = np.empty(shape, dtype=np.float64)

    for row, imag in enumerate(y):
        for column, real in enumerate(x):
            result = _diagnostic(
                a,
                eigenvalues,
                complex(real, imag),
            )

            sigma[row, column] = result.smallest_singular_value
            distances[row, column] = result.spectral_distance
            amplification[row, column] = result.amplification_ratio

    return PseudospectralGrid(
        real_axis=x.copy(),
        imaginary_axis=y.copy(),
        smallest_singular_values=sigma,
        spectral_distances=distances,
        amplification_ratios=amplification,
    )


def pseudospectral_membership(
    smallest_singular_values: ArrayLike,
    epsilon: float,
) -> NDArray[np.bool_]:
    """Classify points in the open epsilon-pseudospectrum.

    Membership uses sigma_min < epsilon. Numerical boundary
    classification remains sensitive to singular-value error.
    """
    values = np.asarray(
        smallest_singular_values,
        dtype=np.float64,
    )

    if np.any(np.isnan(values)) or np.any(values < 0):
        raise ValueError(
            "smallest_singular_values must be nonnegative and not NaN"
        )

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")

    return np.asarray(values < epsilon, dtype=np.bool_)
