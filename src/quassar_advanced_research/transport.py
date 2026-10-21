"""Conservative finite-volume solver for periodic advection-diffusion."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TransportParameters:
    velocity: float
    diffusivity: float
    length: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.velocity):
            raise ValueError("Velocity must be finite")
        if not np.isfinite(self.diffusivity) or self.diffusivity < 0:
            raise ValueError("Diffusivity must be finite and nonnegative")
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("Domain length must be finite and positive")


def stable_timestep(dx, velocity, diffusivity, safety=0.8):
    """Conservative explicit Euler stability bound."""
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    if not np.isfinite(velocity):
        raise ValueError("velocity must be finite")
    if not np.isfinite(diffusivity) or diffusivity < 0:
        raise ValueError("diffusivity must be finite and nonnegative")
    if not 0 < safety <= 1:
        raise ValueError("safety must lie in (0, 1]")
    rate = abs(velocity) / dx + 2 * diffusivity / dx**2
    return safety / rate if rate > 0 else np.inf


def rhs(u, params):
    """Cell-average evolution from conservative interface fluxes."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size < 3 or not np.all(np.isfinite(u)):
        raise ValueError("u must be a finite 1D array of >= 3 cells")

    dx = params.length / u.size
    right = np.roll(u, -1)

    advective = params.velocity * (
        u if params.velocity >= 0 else right
    )
    diffusive = -params.diffusivity * (right - u) / dx
    flux = advective + diffusive

    return -(flux - np.roll(flux, 1)) / dx


def solve(u0, params, final_time, safety=0.8):
    """Integrate periodic cell averages with stable explicit Euler."""
    u = np.asarray(u0, dtype=float).copy()
    if u.ndim != 1 or u.size < 3 or not np.all(np.isfinite(u)):
        raise ValueError("Invalid initial state")
    if not np.isfinite(final_time) or final_time < 0:
        raise ValueError("final_time must be finite and nonnegative")

    dx = params.length / u.size
    dt_limit = stable_timestep(
        dx, params.velocity, params.diffusivity, safety
    )

    if final_time == 0:
        return u

    steps = max(1, int(np.ceil(final_time / dt_limit)))
    dt = final_time / steps

    for _ in range(steps):
        u += dt * rhs(u, params)

    return u
