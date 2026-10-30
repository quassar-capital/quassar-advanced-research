
"""Spin-weighted spheroidal angular eigenvalue solver for Kerr.

Uses a Galerkin expansion in spin-weighted spherical harmonics.
The Fourier convention is exp(-i omega t + i m phi).

The angular separation eigenvalue A is defined by

    S'' + cot(theta) S'
    + [c**2 cos(theta)**2 - 2*c*s*cos(theta)
       - (m+s*cos(theta))**2/sin(theta)**2
       + s + A] S = 0,

where c = a*omega.

Finite-dimensional truncation differences are convergence diagnostics,
not rigorous spectral error certificates. Mode identification is local
to the spherical reference and is not global analytic continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SPIN_WEIGHT = -2


@dataclass(frozen=True)
class AngularEigenvalueResult:
    spin_weight: int
    ell: int
    azimuthal_number: int
    spheroidicity: complex
    eigenvalue: complex
    truncation_ell: int
    convergence_difference: float
    relative_convergence_difference: float
    eigenpair_residual: float
    reference_mode_overlap: float


def _validate_quantum_numbers(
    ell: int,
    m: int,
) -> None:
    if (
        not isinstance(ell, (int, np.integer))
        or isinstance(ell, (bool, np.bool_))
        or not isinstance(m, (int, np.integer))
        or isinstance(m, (bool, np.bool_))
    ):
        raise ValueError("ell and m must be integers")

    if ell < max(abs(m), abs(SPIN_WEIGHT)):
        raise ValueError("ell must be at least max(abs(m), 2)")


def _cosine_matrix(
    m: int,
    ell_max: int,
) -> NDArray[np.float64]:
    """Matrix of cos(theta) in the spin-weighted spherical basis."""
    s = SPIN_WEIGHT
    ell_min = max(abs(m), abs(s))
    degrees = np.arange(ell_min, ell_max + 1)
    size = len(degrees)

    matrix = np.zeros((size, size), dtype=np.float64)

    for index, ell in enumerate(degrees):
        matrix[index, index] = -s * m / (ell * (ell + 1))

        if index + 1 < size:
            upper = ell + 1
            numerator = (
                (upper**2 - m**2)
                * (upper**2 - s**2)
            )
            denominator = (
                upper * np.sqrt((2 * ell + 1) * (2 * ell + 3))
            )
            coefficient = np.sqrt(float(numerator)) / denominator
            matrix[index, index + 1] = coefficient
            matrix[index + 1, index] = coefficient

    return matrix


def _angular_matrix(
    m: int,
    c: complex,
    ell_max: int,
) -> NDArray[np.complex128]:
    """Construct the Galerkin angular operator.

    One additional basis state is retained when forming C**2, so
    the highest retained mode includes its intermediate coupling.
    """
    s = SPIN_WEIGHT
    ell_min = max(abs(m), abs(s))
    degrees = np.arange(ell_min, ell_max + 1)
    size = len(degrees)

    extended_cosine = _cosine_matrix(m, ell_max + 1)
    cosine = extended_cosine[:size, :size]
    cosine_squared = (
        extended_cosine @ extended_cosine
    )[:size, :size]

    spherical = degrees * (degrees + 1) - s * (s + 1)

    return (
        np.diag(spherical.astype(np.complex128))
        + 2.0 * s * c * cosine
        - c**2 * cosine_squared
    )


def _selected_eigenpair(
    matrix: NDArray[np.complex128],
    ell: int,
    m: int,
) -> tuple[complex, float, float]:
    """Select the eigenpair nearest the spherical reference eigenvalue."""
    ell_min = max(abs(m), abs(SPIN_WEIGHT))
    reference_index = ell - ell_min
    reference_value = (
        ell * (ell + 1)
        - SPIN_WEIGHT * (SPIN_WEIGHT + 1)
    )

    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values - reference_value)))

    eigenvalue = complex(values[index])
    vector = vectors[:, index]
    vector_norm = float(np.linalg.norm(vector))

    if vector_norm == 0.0:
        raise ArithmeticError("eigenvector has zero numerical norm")

    residual = float(
        np.linalg.norm(matrix @ vector - eigenvalue * vector)
        / (max(1.0, np.linalg.norm(matrix, 2)) * vector_norm)
    )

    overlap = float(
        abs(vector[reference_index]) / vector_norm
    )

    return eigenvalue, residual, overlap


def gravitational_angular_eigenvalue(
    ell: int,
    azimuthal_number: int,
    spheroidicity: complex,
    *,
    truncation_ell: int | None = None,
    convergence_extension: int = 4,
) -> AngularEigenvalueResult:
    """Compute a locally identified gravitational angular eigenvalue.

    The selected eigenvalue is nearest the c=0 spherical reference.
    This is appropriate for a local continuation regime; large |c|,
    near-degeneracies, and mode crossings require explicit continuation
    and additional spectral-separation analysis.

    The reported convergence difference compares ell_max and
    ell_max + convergence_extension.
    """
    m = azimuthal_number
    _validate_quantum_numbers(ell, m)

    c = complex(spheroidicity)
    if not np.isfinite(c.real) or not np.isfinite(c.imag):
        raise ValueError("spheroidicity must be finite")

    ell_min = max(abs(m), abs(SPIN_WEIGHT))

    if truncation_ell is None:
        truncation_ell = max(ell + 8, ell_min + 10)

    if (
        not isinstance(truncation_ell, (int, np.integer))
        or isinstance(truncation_ell, (bool, np.bool_))
        or truncation_ell < ell
    ):
        raise ValueError("truncation_ell must be an integer >= ell")

    if (
        not isinstance(convergence_extension, (int, np.integer))
        or isinstance(convergence_extension, (bool, np.bool_))
        or convergence_extension < 1
    ):
        raise ValueError("convergence_extension must be positive")

    coarse_matrix = _angular_matrix(m, c, truncation_ell)
    fine_matrix = _angular_matrix(
        m, c, truncation_ell + convergence_extension
    )

    coarse_value, _, _ = _selected_eigenpair(
        coarse_matrix, ell, m
    )
    fine_value, residual, overlap = _selected_eigenpair(
        fine_matrix, ell, m
    )

    difference = float(abs(fine_value - coarse_value))
    relative_difference = difference / max(1.0, abs(fine_value))

    return AngularEigenvalueResult(
        spin_weight=SPIN_WEIGHT,
        ell=int(ell),
        azimuthal_number=int(m),
        spheroidicity=c,
        eigenvalue=fine_value,
        truncation_ell=int(
            truncation_ell + convergence_extension
        ),
        convergence_difference=difference,
        relative_convergence_difference=relative_difference,
        eigenpair_residual=residual,
        reference_mode_overlap=overlap,
    )
