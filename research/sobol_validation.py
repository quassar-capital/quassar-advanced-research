"""Independent-replication validation of Sobol sensitivity estimates."""

import numpy as np

from uncertainty import sobol_indices


def validate(replicates=20, samples=50_000, seed=2026):
    first = []
    total = []

    for i in range(replicates):
        s, st = sobol_indices(samples, seed + i * 1009)
        first.append(s)
        total.append(st)

    first = np.asarray(first)
    total = np.asarray(total)

    names = ("Velocity", "Diffusivity")

    print("SOBOL ESTIMATOR VALIDATION")
    print(f"Independent replications: {replicates}")
    print(f"Samples per replication: {samples}")

    for j, name in enumerate(names):
        for label, estimates in (
            ("First-order", first[:, j]),
            ("Total-order", total[:, j]),
        ):
            mean = estimates.mean()
            se = estimates.std(ddof=1) / np.sqrt(replicates)
            lower = mean - 1.96 * se
            upper = mean + 1.96 * se

            print(
                f"{name} {label}: mean={mean:.6f}, "
                f"SE={se:.6f}, "
                f"approx. 95% CI=[{lower:.6f}, {upper:.6f}]"
            )

    sum_first = first.sum(axis=1)
    print(
        f"First-order sum: mean={sum_first.mean():.6f}, "
        f"SD={sum_first.std(ddof=1):.6f}"
    )

    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(total))
    assert np.all(np.abs(first.mean(axis=0) - total.mean(axis=0)) < 0.03)
    assert abs(sum_first.mean() - 1.0) < 0.03

    print("RESULT: PASS — independent-replication validation")


if __name__ == "__main__":
    validate()
