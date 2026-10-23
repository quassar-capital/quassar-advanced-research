"""Conservative Euler equations and HLL flux for one-dimensional ideal-gas flow."""

import math
from typing import Tuple

State = Tuple[float, float, float]


def validate_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    return gamma


def validate_state(state: State, gamma: float) -> float:
    gamma = validate_gamma(gamma)

    if len(state) != 3 or not all(math.isfinite(x) for x in state):
        raise ValueError("Nonfinite or malformed conservative state")

    rho, momentum, energy = state

    if rho <= 0.0:
        raise ValueError("Density must be positive")

    pressure = (gamma - 1.0) * (
        energy - 0.5 * momentum * momentum / rho
    )

    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("Pressure must be positive")

    return pressure


def conservative(
    rho: float, velocity: float, pressure: float, gamma: float
) -> State:
    gamma = validate_gamma(gamma)

    if not all(math.isfinite(x) for x in (rho, velocity, pressure)):
        raise ValueError("Nonfinite primitive state")

    if rho <= 0.0 or pressure <= 0.0:
        raise ValueError("Invalid density or pressure")

    state = (
        rho,
        rho * velocity,
        pressure / (gamma - 1.0)
        + 0.5 * rho * velocity * velocity,
    )

    validate_state(state, gamma)
    return state


def physical_flux(state: State, gamma: float) -> State:
    pressure = validate_state(state, gamma)
    rho, momentum, energy = state
    velocity = momentum / rho

    flux = (
        momentum,
        momentum * velocity + pressure,
        velocity * (energy + pressure),
    )

    if not all(math.isfinite(x) for x in flux):
        raise ValueError("Nonfinite physical flux")

    return flux


def hll_flux(left: State, right: State, gamma: float) -> State:
    p_left = validate_state(left, gamma)
    p_right = validate_state(right, gamma)

    u_left = left[1] / left[0]
    u_right = right[1] / right[0]

    a_left = math.sqrt(gamma * p_left / left[0])
    a_right = math.sqrt(gamma * p_right / right[0])

    s_left = min(u_left - a_left, u_right - a_right)
    s_right = max(u_left + a_left, u_right + a_right)

    if not all(math.isfinite(x) for x in (s_left, s_right)):
        raise ValueError("Nonfinite wave speed")

    f_left = physical_flux(left, gamma)
    f_right = physical_flux(right, gamma)

    if s_left >= 0.0:
        return f_left

    if s_right <= 0.0:
        return f_right

    flux = tuple(
        (
            s_right * f_left[k]
            - s_left * f_right[k]
            + s_left * s_right * (right[k] - left[k])
        ) / (s_right - s_left)
        for k in range(3)
    )

    if not all(math.isfinite(x) for x in flux):
        raise ValueError("Nonfinite HLL flux")

    return flux
