"""Reproducible uncertainty propagation and Sobol sensitivity analysis."""

import numpy as np


def observable(velocity, diffusivity, x=0.25, t=0.1, amplitude=0.2):
    k = 2 * np.pi
    return 1 + amplitude * np.exp(-diffusivity * k**2 * t) * np.sin(
        k * (x - velocity * t)
    )


def sobol_indices(n, seed):
    rng = np.random.default_rng(seed)

    a = rng.random((n, 2))
    b = rng.random((n, 2))

    def evaluate(samples):
        velocity = 0.8 + 0.4 * samples[:, 0]
        diffusivity = 0.01 + 0.02 * samples[:, 1]
        return observable(velocity, diffusivity)

    ya = evaluate(a)
    yb = evaluate(b)

    variance = np.var(np.concatenate((ya, yb)), ddof=1)
    if variance <= 0:
        raise ValueError("Output variance must be positive")

    first_order = []
    total_order = []

    for j in range(2):
        ab = a.copy()
        ab[:, j] = b[:, j]
        yab = evaluate(ab)

        # Jansen total-order estimator.
        total = np.mean((ya - yab) ** 2) / (2 * variance)

        # Complementary first-order estimator.
        first = 1 - np.mean((yb - yab) ** 2) / (2 * variance)

        first_order.append(first)
        total_order.append(total)

    return np.array(first_order), np.array(total_order)


def experiment(n=100_000, seed=2026):
    rng = np.random.default_rng(seed)
    velocity = rng.uniform(0.8, 1.2, n)
    diffusivity = rng.uniform(0.01, 0.03, n)
    values = observable(velocity, diffusivity)

    first, total = sobol_indices(n, seed + 1)

    print("UNCERTAINTY QUANTIFICATION — CORRECTED")
    print(f"Monte Carlo samples: {n}")
    print(f"Mean observable: {values.mean():.10f}")
    print(f"Standard deviation: {values.std(ddof=1):.10f}")

    lower, upper = np.quantile(values, [0.025, 0.975])
    print(f"95% output quantile interval: [{lower:.10f}, {upper:.10f}]")

    for name, s, st in zip(
        ("Velocity", "Diffusivity"), first, total
    ):
        print(f"{name}: first-order={s:.6f}, total-order={st:.6f}")

    print(f"Sum of first-order indices: {first.sum():.6f}")

    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(total))
    assert np.all(first > -0.03)
    assert np.all(total >= 0)
    assert abs(first.sum() - 1) < 0.05

    print("RESULT: PASS — corrected sensitivity estimators executed")


if __name__ == "__main__":
    experiment()
