
"""Local continuation of spin-minus-two Kerr angular eigenpairs.

Tracks finite-dimensional Galerkin eigenpairs from the spherical limit.
Overlap, separation and step-size checks are numerical safeguards, not
proofs of global analytic continuation or physical mode identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from angular_solver import (
    SPIN_WEIGHT,
    _angular_matrix,
    _validate_quantum_numbers,
)


@dataclass(frozen=True)
class ModeContinuationStep:
    spheroidicity: complex
    eigenvalue: complex
    overlap: float
    overlap_margin: float
    spectral_gap: float
    relative_residual: float
    displacement_fraction: float


@dataclass(frozen=True)
class ModeContinuationResult:
    spin_weight: int
    ell: int
    azimuthal_number: int
    truncation_ell: int
    steps: tuple[ModeContinuationStep, ...]


def continue_gravitational_angular_mode(
    ell: int,
    azimuthal_number: int,
    spheroidicity_path: Sequence[complex],
    *,
    truncation_ell: int | None = None,
    minimum_overlap: float = 0.75,
    minimum_overlap_margin: float = 0.10,
    maximum_displacement_fraction: float = 0.40,
    separation_tolerance: float = 1e-10,
) -> ModeContinuationResult:
    """Track one locally identifiable Galerkin eigenpair along a supplied path.

    The path must begin at c=0 and include at least one subsequent point.
    Eigenvector overlaps use the Euclidean inner product. Step acceptance
    is deliberately conservative; rejection does not establish a physical
    degeneracy. No automatic path refinement or global branch guarantee is
    implied. The caller must check convergence across truncations.
    """
    m = azimuthal_number
    _validate_quantum_numbers(ell, m)

    if truncation_ell is None:
        truncation_ell = max(
            ell + 8,
            max(abs(m), abs(SPIN_WEIGHT)) + 10,
        )

    if (
        not isinstance(truncation_ell, (int, np.integer))
        or isinstance(truncation_ell, (bool, np.bool_))
        or truncation_ell < ell
    ):
        raise ValueError("truncation_ell must be an integer >= ell")

    limits = (
        (minimum_overlap, 0.0, 1.0),
        (minimum_overlap_margin, 0.0, 1.0),
        (maximum_displacement_fraction, 0.0, 1.0),
        (separation_tolerance, 0.0, 1.0),
    )

    if any(
        not np.isfinite(v) or not lo < v < hi
        for v, lo, hi in limits
    ):
        raise ValueError(
            "all numerical thresholds must lie strictly between 0 and 1"
        )

    try:
        path = tuple(complex(c) for c in spheroidicity_path)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "spheroidicity_path must contain complex numbers"
        ) from exc

    if len(path) < 2 or path[0] != 0j:
        raise ValueError(
            "path must start at c=0 and contain at least two points"
        )

    if any(
        not np.isfinite(c.real) or not np.isfinite(c.imag)
        for c in path
    ):
        raise ValueError(
            "path must contain finite complex values"
        )

    if any(a == b for a, b in zip(path, path[1:])):
        raise ValueError(
            "consecutive path points must be distinct"
        )

    reference_index = ell - max(abs(m), abs(SPIN_WEIGHT))

    previous_vector = None
    previous_value = None
    previous_gap = None

    steps: list[ModeContinuationStep] = []

    for step_index, c in enumerate(path):
        matrix = _angular_matrix(
            m,
            c,
            int(truncation_ell),
        )

        eigenvalues, vectors = np.linalg.eig(matrix)

        if (
            not np.all(np.isfinite(eigenvalues))
            or not np.all(np.isfinite(vectors))
        ):
            raise ArithmeticError(
                f"nonfinite eigenpair at path index {step_index}"
            )

        vectors /= np.linalg.norm(vectors, axis=0)

        if step_index == 0:
            selected = reference_index
            overlap = 1.0
            margin = 1.0
            displacement = 0.0

        else:
            overlaps = np.abs(
                previous_vector.conj() @ vectors
            )

            order = np.argsort(
                -overlaps,
                kind="stable",
            )

            selected = int(order[0])
            overlap = float(overlaps[selected])

            runner_up = (
                float(overlaps[order[1]])
                if len(order) > 1
                else 0.0
            )

            margin = overlap - runner_up

            if (
                overlap < minimum_overlap
                or margin < minimum_overlap_margin
            ):
                raise ArithmeticError(
                    "ambiguous eigenvector identification "
                    f"at path index {step_index}"
                )

            displacement = float(
                abs(
                    eigenvalues[selected]
                    - previous_value
                )
                / previous_gap
            )

            if (
                displacement
                > maximum_displacement_fraction
            ):
                raise ArithmeticError(
                    "path step too large relative to "
                    "previous spectral gap "
                    f"at index {step_index}"
                )

        value = complex(eigenvalues[selected])
        vector = vectors[:, selected]

        other = np.delete(
            eigenvalues,
            selected,
        )

        gap = (
            float(np.min(np.abs(other - value)))
            if other.size
            else float("inf")
        )

        scale = max(
            1.0,
            float(np.linalg.norm(matrix, ord=2)),
            abs(value),
        )

        if gap <= separation_tolerance * scale:
            raise ArithmeticError(
                "insufficient spectral separation "
                f"at path index {step_index}"
            )

        residual = float(
            np.linalg.norm(
                matrix @ vector - value * vector
            )
            / scale
        )

        if not np.isfinite(residual):
            raise ArithmeticError(
                "nonfinite eigenpair residual "
                f"at path index {step_index}"
            )

        steps.append(
            ModeContinuationStep(
                spheroidicity=c,
                eigenvalue=value,
                overlap=overlap,
                overlap_margin=margin,
                spectral_gap=gap,
                relative_residual=residual,
                displacement_fraction=displacement,
            )
        )

        previous_vector = vector
        previous_value = value
        previous_gap = gap

    return ModeContinuationResult(
        spin_weight=SPIN_WEIGHT,
        ell=int(ell),
        azimuthal_number=int(m),
        truncation_ell=int(truncation_ell),
        steps=tuple(steps),
    )
