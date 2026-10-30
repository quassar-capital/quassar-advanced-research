
"""Analytical verification of Research 04 pseudospectral diagnostics.

The benchmarks test finite-dimensional Euclidean resolvent behavior.
They do not establish continuous-operator Kerr pseudospectra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from pseudospectrum import (
    pseudospectral_grid,
    pseudospectral_membership,
    resolvent_diagnostic,
)


def test_normal_operator_resolvent_identity():
    """For normal matrices, sigma_min equals spectral distance."""
    operator = np.diag([1.0, 3.0]).astype(complex)
    z = 2.0 + 1.0j

    result = resolvent_diagnostic(operator, z)

    expected_distance = np.sqrt(2.0)

    assert result.smallest_singular_value == pytest.approx(
        expected_distance, rel=1e-12
    )
    assert result.spectral_distance == pytest.approx(
        expected_distance, rel=1e-12
    )
    assert result.resolvent_norm == pytest.approx(
        1.0 / expected_distance, rel=1e-12
    )
    assert result.amplification_ratio == pytest.approx(
        1.0, rel=1e-12
    )


def test_nonnormal_operator_exhibits_resolvent_amplification():
    """A triangular non-normal operator has an exact resolvent benchmark."""
    operator = np.array(
        [[1.0, 10.0], [0.0, 2.0]],
        dtype=complex,
    )
    z = 1.5 + 0.0j

    result = resolvent_diagnostic(operator, z)

    # zI - A = [[0.5, -10], [0, -0.5]].
    # Its singular values have product 0.25.
    # The larger singular value is determined by its Frobenius norm.
    frobenius_squared = 100.5
    discriminant = np.sqrt(frobenius_squared**2 - 4.0 * 0.25**2)
    largest_squared = 0.5 * (
        frobenius_squared + discriminant
    )
    expected_sigma_min = 0.25 / np.sqrt(largest_squared)

    assert result.spectral_distance == pytest.approx(
        0.5, rel=1e-12
    )
    assert result.smallest_singular_value == pytest.approx(
        expected_sigma_min, rel=1e-12
    )
    assert result.resolvent_norm == pytest.approx(
        1.0 / expected_sigma_min, rel=1e-12
    )
    assert result.amplification_ratio == pytest.approx(
        0.5 / expected_sigma_min, rel=1e-12
    )
    assert result.amplification_ratio > 10.0


def test_complex_grid_matches_pointwise_diagnostics():
    """Grid orientation and numerical values must match pointwise results."""
    operator = np.array(
        [[1.0, 2.0], [0.0, 3.0]],
        dtype=complex,
    )
    real_axis = np.array([0.5, 1.5, 2.5])
    imaginary_axis = np.array([-1.0, 0.5])

    grid = pseudospectral_grid(
        operator,
        real_axis,
        imaginary_axis,
    )

    assert grid.smallest_singular_values.shape == (2, 3)
    assert grid.spectral_distances.shape == (2, 3)
    assert grid.amplification_ratios.shape == (2, 3)

    for row, imaginary in enumerate(imaginary_axis):
        for column, real in enumerate(real_axis):
            point = resolvent_diagnostic(
                operator,
                complex(real, imaginary),
            )

            assert grid.smallest_singular_values[
                row, column
            ] == pytest.approx(
                point.smallest_singular_value, rel=1e-12
            )
            assert grid.spectral_distances[
                row, column
            ] == pytest.approx(
                point.spectral_distance, rel=1e-12
            )
            assert grid.amplification_ratios[
                row, column
            ] == pytest.approx(
                point.amplification_ratio, rel=1e-12
            )


def test_epsilon_pseudospectrum_strict_boundary():
    """Membership must respect the strict singular-value threshold."""
    singular_values = np.array(
        [[0.05, 0.10, 0.20], [0.0, 0.15, 0.30]],
        dtype=float,
    )

    membership = pseudospectral_membership(
        singular_values,
        epsilon=0.10,
    )

    expected = np.array(
        [[True, False, False], [True, False, False]],
        dtype=bool,
    )

    np.testing.assert_array_equal(membership, expected)

    with pytest.raises(ValueError):
        pseudospectral_membership(singular_values, epsilon=0.0)

    with pytest.raises(ValueError):
        pseudospectral_membership(
            np.array([0.01, np.nan]),
            epsilon=0.10,
        )
