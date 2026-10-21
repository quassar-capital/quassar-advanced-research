"""Verification tests for uncertainty propagation and Sobol estimators."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from uncertainty import observable, sobol_indices


def test_observable_is_finite():
    velocity = np.linspace(0.8, 1.2, 100)
    diffusivity = np.linspace(0.01, 0.03, 100)
    result = observable(velocity, diffusivity)

    assert result.shape == (100,)
    assert np.all(np.isfinite(result))


def test_sobol_reproducibility():
    first_a, total_a = sobol_indices(10_000, seed=42)
    first_b, total_b = sobol_indices(10_000, seed=42)

    assert np.array_equal(first_a, first_b)
    assert np.array_equal(total_a, total_b)


def test_sobol_physical_consistency():
    first, total = sobol_indices(50_000, seed=2026)

    assert np.all(first >= -0.03)
    assert np.all(first <= 1.03)
    assert np.all(total >= 0)
    assert np.all(total <= 1.03)
    assert abs(first.sum() - 1) < 0.04
    assert np.all(np.abs(first - total) < 0.04)


def test_velocity_dominates_for_selected_observable():
    first, total = sobol_indices(50_000, seed=123)

    assert first[0] > first[1]
    assert total[0] > total[1]


def test_zero_amplitude_has_zero_variance():
    values = observable(
        np.array([0.8, 1.0, 1.2]),
        np.array([0.01, 0.02, 0.03]),
        amplitude=0.0,
    )
    assert np.array_equal(values, np.ones(3))
