"""Analytical verification of Research 04 structured spectral sensitivity.

These tests establish finite-dimensional mathematical correctness.
They do not assert physical admissibility of Kerr perturbations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from structured import structured_eigenvalue_sensitivity


def test_diagonal_perturbation_exact_reference():
    """A diagonal restriction gives an exact first-order sensitivity."""
    operator = np.diag([1.0, 3.0]).astype(complex)
    basis = [
        np.array(
            [[1.0, 0.0], [0.0, 0.0]],
            dtype=complex,
        )
    ]

    result = structured_eigenvalue_sensitivity(
        operator, 0, basis
    )

    assert result.eigenvalue == pytest.approx(1.0)
    assert result.structured_bound == pytest.approx(1.0)
    assert result.unstructured_bound == pytest.approx(1.0)
    assert result.relative_sensitivity == pytest.approx(1.0)
    assert abs(result.directional_derivative) == pytest.approx(1.0)
    assert np.linalg.norm(result.perturbation, "fro") == pytest.approx(1.0)


def test_restricted_perturbation_zero_sensitivity():
    """An off-diagonal restriction cannot shift this eigenvalue at first order."""
    operator = np.diag([1.0, 3.0]).astype(complex)
    basis = [
        np.array(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=complex,
        )
    ]

    result = structured_eigenvalue_sensitivity(
        operator, 0, basis
    )

    assert result.structured_bound == pytest.approx(0.0, abs=1e-13)
    assert result.relative_sensitivity == pytest.approx(0.0, abs=1e-13)
    assert result.directional_derivative == pytest.approx(0.0, abs=1e-13)
    assert np.linalg.norm(result.perturbation, "fro") == pytest.approx(
        0.0, abs=1e-13
    )


def test_complete_basis_recovers_unstructured_bound():
    """The full complex matrix space must recover unrestricted sensitivity."""
    operator = np.array(
        [[1.0, 10.0], [0.0, 2.0]],
        dtype=complex,
    )

    basis = []
    for i in range(2):
        for j in range(2):
            matrix = np.zeros((2, 2), dtype=complex)
            matrix[i, j] = 1.0
            basis.append(matrix)

    result = structured_eigenvalue_sensitivity(
        operator, 0, basis
    )

    expected = np.sqrt(101.0)

    assert result.basis_rank == 4
    assert result.unstructured_bound == pytest.approx(
        expected, rel=1e-12
    )
    assert result.structured_bound == pytest.approx(
        expected, rel=1e-12
    )
    assert result.relative_sensitivity == pytest.approx(
        1.0, rel=1e-12
    )
    assert abs(result.directional_derivative) == pytest.approx(
        expected, rel=1e-12
    )
    assert np.linalg.norm(result.perturbation, "fro") == pytest.approx(
        1.0, rel=1e-12
    )


def test_basis_scaling_and_redundancy_invariance():
    """Equivalent complex subspaces must produce the same sensitivity."""
    operator = np.array(
        [[1.0, 4.0], [0.0, 2.0]],
        dtype=complex,
    )

    b1 = np.array(
        [[1.0, 0.0], [0.0, 0.0]],
        dtype=complex,
    )
    b2 = np.array(
        [[0.0, 0.0], [1.0, 0.0]],
        dtype=complex,
    )

    reference = structured_eigenvalue_sensitivity(
        operator, 0, [b1, b2]
    )

    transformed = structured_eigenvalue_sensitivity(
        operator,
        0,
        [
            3.0 * b1,
            2.0j * b2,
            b1 + b2,
        ],
    )

    assert reference.basis_rank == 2
    assert transformed.basis_rank == 2
    assert transformed.structured_bound == pytest.approx(
        reference.structured_bound, rel=1e-12
    )
    assert transformed.relative_sensitivity == pytest.approx(
        reference.relative_sensitivity, rel=1e-12
    )
    assert abs(transformed.directional_derivative) == pytest.approx(
        abs(reference.directional_derivative), rel=1e-12
    )
