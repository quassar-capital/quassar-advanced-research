
"""Independent numerical checks for Research 04 spectral diagnostics.

The reference matrices are analytically tractable and are not models of
Kerr spacetime. These tests verify finite-dimensional mathematical
identities before physical operator discretization is introduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from spectral import eigenvalue_sensitivity, smallest_singular_value


def test_resolvent_singular_value_against_exact_reference():
    """A diagonal normal matrix has analytically known singular values."""
    operator = np.diag([1.0, 3.0]).astype(complex)
    z = 2.0 + 0.5j

    expected = min(abs(z - 1.0), abs(z - 3.0))
    computed = smallest_singular_value(operator, z)

    assert computed == pytest.approx(expected, rel=1e-13, abs=1e-14)


def test_normal_eigenvalue_conditioning():
    """A simple eigenvalue of a normal matrix has condition number one."""
    operator = np.diag([1.0, 2.0]).astype(complex)
    perturbation = np.diag([3.0, -4.0]).astype(complex)

    result = eigenvalue_sensitivity(operator, perturbation, 0)

    assert result.eigenvalue == pytest.approx(1.0)
    assert result.condition_number == pytest.approx(1.0, abs=1e-13)
    assert result.directional_derivative == pytest.approx(3.0)
    assert result.right_residual < 1e-13
    assert result.left_residual < 1e-13


def test_non_normal_eigenvalue_sensitivity():
    """An upper-triangular non-normal reference has an exact derivative."""
    operator = np.array(
        [[1.0, 10.0], [0.0, 2.0]],
        dtype=complex,
    )
    perturbation = np.array(
        [[0.0, 0.0], [1.0, 0.0]],
        dtype=complex,
    )

    result = eigenvalue_sensitivity(operator, perturbation, 0)

    assert result.eigenvalue == pytest.approx(1.0)
    assert result.directional_derivative == pytest.approx(-10.0)
    assert result.condition_number == pytest.approx(
        np.sqrt(101.0), rel=1e-12
    )
    assert result.right_residual < 1e-12
    assert result.left_residual < 1e-12


def test_directional_derivative_against_finite_difference():
    """Centered differences converge to the analytic eigenvalue derivative."""
    operator = np.array(
        [[1.0, 10.0], [0.0, 2.0]],
        dtype=complex,
    )
    perturbation = np.array(
        [[0.0, 0.0], [1.0, 0.0]],
        dtype=complex,
    )

    analytic = eigenvalue_sensitivity(
        operator, perturbation, 0
    ).directional_derivative

    def selected_eigenvalue(eta: float) -> complex:
        values = np.linalg.eigvals(operator + eta * perturbation)
        return complex(values[np.argmin(abs(values - 1.0))])

    h_coarse = 1e-3
    h_fine = 1e-4

    coarse = (
        selected_eigenvalue(h_coarse)
        - selected_eigenvalue(-h_coarse)
    ) / (2.0 * h_coarse)

    fine = (
        selected_eigenvalue(h_fine)
        - selected_eigenvalue(-h_fine)
    ) / (2.0 * h_fine)

    coarse_error = abs(coarse - analytic)
    fine_error = abs(fine - analytic)

    assert fine_error < coarse_error
    assert fine_error < 1e-4
