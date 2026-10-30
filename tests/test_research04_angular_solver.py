
"""Analytical verification of the gravitational angular eigenvalue solver.

Independent references include exact spin-weighted spherical
eigenvalues, first-order spheroidal perturbation theory, and
complex-conjugation symmetry.

Truncation agreement is a numerical convergence diagnostic, not
a rigorous eigenvalue enclosure or proof of global mode tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from angular_solver import gravitational_angular_eigenvalue


def test_exact_spin_weighted_spherical_limit():
    """At c=0, the angular eigenvalues are known exactly."""
    for ell, m in [(2, 2), (3, 1), (4, -2)]:
        result = gravitational_angular_eigenvalue(
            ell,
            m,
            0.0,
            truncation_ell=ell + 6,
        )

        expected = ell * (ell + 1) - 2

        assert result.eigenvalue == pytest.approx(
            expected, rel=0.0, abs=1e-12
        )
        assert result.convergence_difference == pytest.approx(
            0.0, abs=1e-12
        )
        assert result.eigenpair_residual < 1e-12
        assert result.reference_mode_overlap == pytest.approx(
            1.0, abs=1e-12
        )


def test_first_order_spheroidal_eigenvalue_sensitivity():
    """Check the exact first derivative at c=0."""
    ell = 2
    m = 2
    step = 1e-5

    positive = gravitational_angular_eigenvalue(
        ell,
        m,
        step,
        truncation_ell=10,
    )
    negative = gravitational_angular_eigenvalue(
        ell,
        m,
        -step,
        truncation_ell=10,
    )

    numerical_derivative = (
        positive.eigenvalue - negative.eigenvalue
    ) / (2.0 * step)

    analytical_derivative = (
        -8.0 * m / (ell * (ell + 1))
    )

    assert numerical_derivative.real == pytest.approx(
        analytical_derivative, rel=1e-7, abs=1e-8
    )
    assert numerical_derivative.imag == pytest.approx(
        0.0, abs=1e-10
    )


def test_complex_spheroidicity_conjugation_symmetry():
    """For real angular couplings, A(conj(c)) = conj(A(c))."""
    c = 0.15 + 0.07j

    result = gravitational_angular_eigenvalue(
        3,
        1,
        c,
        truncation_ell=12,
    )
    conjugate_result = gravitational_angular_eigenvalue(
        3,
        1,
        c.conjugate(),
        truncation_ell=12,
    )

    assert conjugate_result.eigenvalue == pytest.approx(
        result.eigenvalue.conjugate(),
        rel=1e-11,
        abs=1e-11,
    )
    assert result.eigenpair_residual < 1e-11
    assert conjugate_result.eigenpair_residual < 1e-11


def test_truncation_convergence_and_mode_identification():
    """Independent truncations must agree for a moderate complex c."""
    c = 0.2 + 0.1j

    lower = gravitational_angular_eigenvalue(
        2,
        2,
        c,
        truncation_ell=8,
        convergence_extension=4,
    )
    higher = gravitational_angular_eigenvalue(
        2,
        2,
        c,
        truncation_ell=12,
        convergence_extension=4,
    )

    assert lower.truncation_ell == 12
    assert higher.truncation_ell == 16

    assert lower.eigenvalue == pytest.approx(
        higher.eigenvalue,
        rel=1e-9,
        abs=1e-10,
    )

    assert lower.convergence_difference < 1e-8
    assert higher.convergence_difference < 1e-8

    assert lower.eigenpair_residual < 1e-11
    assert higher.eigenpair_residual < 1e-11

    assert lower.reference_mode_overlap > 0.9
    assert higher.reference_mode_overlap > 0.9

    assert np.isfinite(lower.relative_convergence_difference)
    assert np.isfinite(higher.relative_convergence_difference)
