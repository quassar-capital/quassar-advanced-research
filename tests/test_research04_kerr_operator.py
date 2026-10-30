
"""Analytical verification of the scalar Kerr perturbation formulation.

Benchmarks cover subextremal horizon geometry, the Schwarzschild
limit, an exact spherical-harmonic solution, and an exact static
radial solution.

These tests do not establish gravitational spin-minus-two mode
spectra or validate numerical Kerr boundary-value discretizations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from kerr_operator import (
    KerrBackground,
    ScalarMode,
    angular_coefficients,
    angular_residual,
    radial_coefficients,
    radial_residual,
    separation_constant,
)


def test_kerr_horizon_geometry_exact_relations():
    """Verify horizon identities and angular velocity."""
    mass = 2.0
    spin = 1.2
    background = KerrBackground(mass=mass, spin=spin)

    expected_gap = np.sqrt(mass**2 - spin**2)
    expected_outer = mass + expected_gap
    expected_inner = mass - expected_gap

    assert background.dimensionless_spin == pytest.approx(0.6)
    assert background.outer_horizon == pytest.approx(expected_outer)
    assert background.inner_horizon == pytest.approx(expected_inner)

    assert (
        background.outer_horizon + background.inner_horizon
    ) == pytest.approx(2.0 * mass)

    assert (
        background.outer_horizon * background.inner_horizon
    ) == pytest.approx(spin**2)

    assert background.horizon_angular_velocity == pytest.approx(
        spin / (2.0 * mass * expected_outer)
    )

    radii = np.array([4.0, 5.0, 8.0])
    expected_delta = radii**2 - 2.0 * mass * radii + spin**2

    np.testing.assert_allclose(
        background.delta(radii),
        expected_delta,
        rtol=1e-13,
        atol=1e-13,
    )

    with pytest.raises(ValueError):
        KerrBackground(mass=1.0, spin=1.0)


def test_schwarzschild_radial_operator_limit():
    """At zero spin, the scalar radial operator has its known form."""
    background = KerrBackground(mass=1.0, spin=0.0)
    mode = ScalarMode(
        azimuthal_number=0,
        frequency=0.4,
        angular_eigenvalue=6.0,
    )

    radii = np.array([3.0, 4.0, 7.0])

    p2, p1, p0 = radial_coefficients(
        background,
        mode,
        radii,
    )

    expected_delta = radii * (radii - 2.0)
    expected_first = 2.0 * (radii - 1.0)
    expected_potential = (
        radii**4 * 0.4**2 / expected_delta - 6.0
    )

    np.testing.assert_allclose(p2, expected_delta, rtol=1e-13)
    np.testing.assert_allclose(p1, expected_first, rtol=1e-13)
    np.testing.assert_allclose(p0, expected_potential, rtol=1e-13)

    assert separation_constant(background, mode) == pytest.approx(6.0)


def test_exact_spherical_harmonic_angular_solution():
    """S(theta) = cos(theta) solves the l=1, m=0 angular equation."""
    background = KerrBackground(mass=1.0, spin=0.0)
    mode = ScalarMode(
        azimuthal_number=0,
        frequency=0.7,
        angular_eigenvalue=2.0,
    )

    theta = np.array([0.3, 0.8, 1.2, 2.0, 2.7])

    value = np.cos(theta)
    first_derivative = -np.sin(theta)
    second_derivative = -np.cos(theta)

    q1, q0 = angular_coefficients(
        background,
        mode,
        theta,
    )

    np.testing.assert_allclose(
        q1,
        np.cos(theta) / np.sin(theta),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        q0,
        np.full_like(theta, 2.0),
        rtol=1e-13,
    )

    residual = angular_residual(
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
        atol=1e-13,
    )


def test_exact_static_scalar_radial_solution():
    """The constant l=m=0, omega=0 scalar mode has zero radial residual."""
    background = KerrBackground(mass=1.0, spin=0.95)
    mode = ScalarMode(
        azimuthal_number=0,
        frequency=0.0,
        angular_eigenvalue=0.0,
    )

    radii = np.array([1.5, 2.0, 3.0, 10.0])

    value = np.ones_like(radii)
    first_derivative = np.zeros_like(radii)
    second_derivative = np.zeros_like(radii)

    residual = radial_residual(
        background,
        mode,
        radii,
        value,
        first_derivative,
        second_derivative,
    )

    np.testing.assert_allclose(
        residual,
        np.zeros_like(radii),
        rtol=0.0,
        atol=1e-13,
    )

    with pytest.raises(ValueError):
        radial_coefficients(
            background,
            mode,
            background.outer_horizon,
        )
