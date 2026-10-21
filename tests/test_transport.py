"""Scientific verification of the conservative transport solver."""

import numpy as np
import pytest

from quassar_advanced_research.transport import (
    TransportParameters,
    rhs,
    solve,
    stable_timestep,
)


def exact_cell_averages(n, t, params, amplitude=0.2):
    """Exact periodic sine-wave cell averages at time t."""
    dx = params.length / n
    centers = (np.arange(n) + 0.5) * dx
    k = 2 * np.pi / params.length
    sinc = np.sin(k * dx / 2) / (k * dx / 2)
    decay = np.exp(-params.diffusivity * k**2 * t)
    return 1 + amplitude * sinc * decay * np.sin(
        k * (centers - params.velocity * t)
    )


@pytest.mark.parametrize("velocity", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("diffusivity", [0.0, 0.01, 0.1])
def test_mass_conservation(velocity, diffusivity):
    p = TransportParameters(velocity, diffusivity)
    x = (np.arange(128) + 0.5) / 128
    initial = 1 + 0.2 * np.sin(2 * np.pi * x)
    final = solve(initial, p, 0.1)
    assert abs(final.mean() - initial.mean()) < 1e-12


def test_constant_solution_is_stationary():
    p = TransportParameters(-3.0, 0.05)
    initial = np.full(64, 2.5)
    assert np.allclose(solve(initial, p, 0.3), initial, atol=1e-12)


def test_rhs_is_conservative():
    p = TransportParameters(1.7, 0.03)
    u = np.random.default_rng(42).normal(size=97)
    assert abs(rhs(u, p).sum()) < 1e-10


def test_discrete_maximum_principle():
    p = TransportParameters(1.0, 0.02)
    rng = np.random.default_rng(123)
    u = rng.uniform(-1, 3, size=80)
    final = solve(u, p, 0.2)
    assert final.min() >= u.min() - 1e-12
    assert final.max() <= u.max() + 1e-12


@pytest.mark.parametrize(
    "velocity,diffusivity",
    [(1.0, 0.0), (0.0, 0.02), (1.0, 0.02), (-1.0, 0.02)],
)
def test_convergence_against_exact_cell_averages(velocity, diffusivity):
    p = TransportParameters(velocity, diffusivity)
    errors = []

    for n in [32, 64, 128]:
        initial = exact_cell_averages(n, 0.0, p)
        numerical = solve(initial, p, 0.1)
        exact = exact_cell_averages(n, 0.1, p)
        errors.append(np.mean(np.abs(numerical - exact)))

    assert errors[2] < errors[1] < errors[0]
    assert errors[0] / errors[2] > 2.5


def test_invalid_physical_parameters():
    with pytest.raises(ValueError):
        TransportParameters(1.0, -0.01)
    with pytest.raises(ValueError):
        TransportParameters(np.nan, 0.01)
    with pytest.raises(ValueError):
        TransportParameters(1.0, 0.01, length=0)


def test_stability_bound():
    dx = 0.01
    dt = stable_timestep(dx, 2.0, 0.03)
    assert dt > 0
    assert dt * (2.0 / dx + 2 * 0.03 / dx**2) <= 1


def test_zero_time_preserves_initial_state():
    p = TransportParameters(1.0, 0.01)
    u = np.arange(20, dtype=float)
    assert np.array_equal(solve(u, p, 0.0), u)
