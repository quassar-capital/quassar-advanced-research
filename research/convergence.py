"""Grid-refinement verification against exact cell averages."""

import numpy as np
from quassar_advanced_research.transport import TransportParameters, solve


def exact_cell_averages(n, t, p, amplitude=0.2):
    dx = p.length / n
    centers = (np.arange(n) + 0.5) * dx
    k = 2 * np.pi / p.length
    cell_factor = np.sinc(dx / p.length)
    return 1 + amplitude * cell_factor * np.exp(
        -p.diffusivity * k**2 * t
    ) * np.sin(k * (centers - p.velocity * t))


def experiment(velocity, diffusivity, final_time=0.1):
    p = TransportParameters(velocity, diffusivity)
    resolutions = [32, 64, 128, 256]
    errors = []

    print(f"\nVelocity={velocity}, diffusivity={diffusivity}")
    print("Cells       L1 error       Observed order")

    for n in resolutions:
        initial = exact_cell_averages(n, 0.0, p)
        numerical = solve(initial, p, final_time)
        exact = exact_cell_averages(n, final_time, p)

        error = np.mean(np.abs(numerical - exact))
        order = (
            np.log2(errors[-1] / error)
            if errors and error > 0
            else float("nan")
        )

        print(f"{n:<8} {error:.8e}    {order:.4f}")
        errors.append(error)

    assert all(
        errors[i + 1] < errors[i]
        for i in range(len(errors) - 1)
    ), "Grid refinement did not reduce the error"

    return errors


if __name__ == "__main__":
    experiment(1.0, 0.0)
    experiment(0.0, 0.02)
    experiment(1.0, 0.02)
    experiment(-1.0, 0.02)
    print("\nRESULT: PASS — errors decrease under grid refinement")
