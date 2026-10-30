
"""Analytical verification of spin-minus-two Teukolsky operators.

Tests use exact differential identities and known separated solutions.
They do not establish Kerr quasinormal frequencies, mode completeness,
or continuous-operator pseudospectral stability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from gravitational_teukolsky import (
    GravitationalMode,
    gravitational_angular_coefficients,
    gravitational_angular_residual,
    gravitational_radial_coefficients,
    gravitational_radial_residual,
    gravitational_separation_constant,
)
from kerr_operator import KerrBackground


def test_schwarzschild_gravitational_radial_coefficients():
    """Check the s=-2 Schwarzschild radial equation independently."""
    background = KerrBackground(mass=1.0, spin=0.0)
    mode = GravitationalMode(
        azimuthal_number=2,
        frequency=0.3,
        angular_eigenvalue=4.0,
    )

    radii = np.array([3.0, 4.0, 7.0])
    delta = radii * (radii - 2.0)
    k = radii**2 * 0.3

    expected_p2 = delta
    expected_p1 = -2.0 * (radii - 1.0)
    expected_p0 = (
        (k**2 + 4.0j * (radii - 1.0) * k) / delta
        - 8.0j * 0.3 * radii
        - 4.0
    )

    p2, p1, p0 = gravitational_radial_coefficients(
        background,
        mode,
        radii,
    )

    np.testing.assert_allclose(p2, expected_p2, rtol=1e-13)
    np.testing.assert_allclose(p1, expected_p1, rtol=1e-13)
    np.testing.assert_allclose(
        p0, expected_p0, rtol=1e-13, atol=1e-13
    )


def test_exact_spin_weighted_angular_solution():
    """The s=-2, l=m=2 harmonic is proportional to (1+cos(theta))**2."""
    background = KerrBackground(mass=1.0, spin=0.0)
    mode = GravitationalMode(
        azimuthal_number=2,
        frequency=0.4,
        angular_eigenvalue=4.0,
    )

    theta = np.array([0.3, 0.7, 1.1, 1.8, 2.5])
    cosine = np.cos(theta)
    sine = np.sin(theta)

    value = (1.0 + cosine) ** 2
    first_derivative = -2.0 * (1.0 + cosine) * sine
    second_derivative = (
        2.0 * sine**2
        - 2.0 * (1.0 + cosine) * cosine
    )

    q1, q0 = gravitational_angular_coefficients(
        background,
        mode,
        theta,
    )

    expected_q0 = (
        -(2.0 - 2.0 * cosine) ** 2 / sine**2
        - 2.0
        + 4.0
    )

    np.testing.assert_allclose(
        q1, cosine / sine, rtol=1e-13
    )
    np.testing.assert_allclose(
        q0, expected_q0, rtol=1e-13
    )

    residual = gravitational_angular_residual(
        background,
        mode,
        theta,
        value,
        first_derivative,
        second_derivative,
    )

    np.testing.assert_allclose(
        residual,
        np.zeros_like(theta),
        rtol=0.0,
        atol=1e-12,
    )


def test_exact_static_gravitational_radial_solution():
    """For omega=m=lambda=0, R=constant solves the radial equation."""
    background = KerrBackground(mass=1.0, spin=0.95)
    mode = GravitationalMode(
        azimuthal_number=0,
        frequency=0.0,
        angular_eigenvalue=0.0,
    )

    radii = np.array([1.5, 2.0, 3.0, 10.0])
    value = np.ones_like(radii)
    zero = np.zeros_like(radii)

    residual = gravitational_radial_residual(
        background,
        mode,
        radii,
        value,
        zero,
        zero,
    )

    np.testing.assert_allclose(
        residual,
        np.zeros_like(radii),
        rtol=0.0,
        atol=1e-13,
    )


def test_gravitational_separation_and_exterior_domain():
    """Verify the lambda convention and reject horizon evaluation."""
    background = KerrBackground(mass=2.0, spin=1.0)
    mode = GravitationalMode(
        azimuthal_number=2,
        frequency=0.25 + 0.10j,
        angular_eigenvalue=4.0 + 0.05j,
    )

    omega = complex(mode.frequency)
    expected_lambda = (
        complex(mode.angular_eigenvalue)
        + background.spin**2 * omega**2
        - 2.0 * background.spin * 2 * omega
    )

    assert gravitational_separation_constant(
        background, mode
    ) == pytest.approx(expected_lambda)

    with pytest.raises(ValueError):
        gravitational_radial_coefficients(
            background,
            mode,
            background.outer_horizon,
        )

    with pytest.raises(ValueError):
        gravitational_angular_coefficients(
            background,
            mode,
            0.0,
        )
