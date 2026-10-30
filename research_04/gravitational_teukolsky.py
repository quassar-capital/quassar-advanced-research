
"""Separated gravitational Teukolsky operators on Kerr backgrounds.

Spin weight s = -2, geometric units G = c = 1, and Fourier convention
exp(-i omega t + i m phi).

The angular eigenvalue is supplied, not solved. The implementation
evaluates differential coefficients and residuals outside the outer
horizon; it does not impose quasinormal-mode boundary conditions or
compute physical spectra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerr_operator import KerrBackground

SPIN_WEIGHT = -2


@dataclass(frozen=True)
class GravitationalMode:
    azimuthal_number: int
    frequency: complex
    angular_eigenvalue: complex

    def __post_init__(self) -> None:
        m = self.azimuthal_number
        if (
            not isinstance(m, (int, np.integer))
            or isinstance(m, (bool, np.bool_))
        ):
            raise ValueError("azimuthal_number must be an integer")

        for name in ("frequency", "angular_eigenvalue"):
            value = complex(getattr(self, name))
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError(f"{name} must be finite")


def _exterior_radius(
    background: KerrBackground,
    radius: ArrayLike,
) -> NDArray[np.float64]:
    r = np.asarray(radius, dtype=np.float64)

    if not np.all(np.isfinite(r)):
        raise ValueError("radius must contain finite values")

    if np.any(r <= background.outer_horizon):
        raise ValueError(
            "radius must lie strictly outside the outer horizon"
        )

    return r


def _interior_angle(
    theta: ArrayLike,
) -> NDArray[np.float64]:
    angle = np.asarray(theta, dtype=np.float64)

    if not np.all(np.isfinite(angle)):
        raise ValueError("theta must contain finite values")

    if np.any((angle <= 0.0) | (angle >= np.pi)):
        raise ValueError("theta must lie strictly between 0 and pi")

    return angle


def gravitational_separation_constant(
    background: KerrBackground,
    mode: GravitationalMode,
) -> complex:
    """Return lambda for the stated spin-weighted angular convention."""
    a = background.spin
    omega = complex(mode.frequency)
    m = mode.azimuthal_number

    return complex(
        mode.angular_eigenvalue
        + a * a * omega * omega
        - 2.0 * a * m * omega
    )


def gravitational_radial_coefficients(
    background: KerrBackground,
    mode: GravitationalMode,
    radius: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """Return p2, p1, p0 for the unscaled spin-minus-two radial ODE.

    The equation is Delta R'' - Delta' R' + V R = 0.
    """
    r = _exterior_radius(background, radius)

    delta = background.delta(r)
    delta_prime = 2.0 * (r - background.mass)

    a = background.spin
    omega = complex(mode.frequency)
    m = mode.azimuthal_number

    k = (r * r + a * a) * omega - a * m
    lam = gravitational_separation_constant(background, mode)

    potential = (
        (k * k + 4.0j * (r - background.mass) * k) / delta
        - 8.0j * omega * r
        - lam
    )

    return (
        np.asarray(delta, dtype=np.float64),
        np.asarray(-delta_prime, dtype=np.float64),
        np.asarray(potential, dtype=np.complex128),
    )


def gravitational_angular_coefficients(
    background: KerrBackground,
    mode: GravitationalMode,
    theta: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """Return q1, q0 for S'' + q1 S' + q0 S = 0."""
    angle = _interior_angle(theta)

    sin_theta = np.sin(angle)
    cos_theta = np.cos(angle)

    a = background.spin
    omega = complex(mode.frequency)
    m = mode.azimuthal_number
    s = SPIN_WEIGHT

    first_derivative = cos_theta / sin_theta

    potential = (
        (a * omega) ** 2 * cos_theta**2
        - 2.0 * a * omega * s * cos_theta
        - (m + s * cos_theta) ** 2 / sin_theta**2
        + s
        + complex(mode.angular_eigenvalue)
    )

    return (
        np.asarray(first_derivative, dtype=np.float64),
        np.asarray(potential, dtype=np.complex128),
    )


def gravitational_radial_residual(
    background: KerrBackground,
    mode: GravitationalMode,
    radius: ArrayLike,
    value: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> NDArray[np.complex128]:
    """Evaluate the gravitational radial differential residual."""
    p2, p1, p0 = gravitational_radial_coefficients(
        background, mode, radius
    )

    r_value = np.asarray(value, dtype=np.complex128)
    r_first = np.asarray(first_derivative, dtype=np.complex128)
    r_second = np.asarray(second_derivative, dtype=np.complex128)

    result = p2 * r_second + p1 * r_first + p0 * r_value

    if not np.all(np.isfinite(result)):
        raise ValueError("radial residual must be finite")

    return np.asarray(result, dtype=np.complex128)


def gravitational_angular_residual(
    background: KerrBackground,
    mode: GravitationalMode,
    theta: ArrayLike,
    value: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> NDArray[np.complex128]:
    """Evaluate the gravitational angular differential residual."""
    q1, q0 = gravitational_angular_coefficients(
        background, mode, theta
    )

    s_value = np.asarray(value, dtype=np.complex128)
    s_first = np.asarray(first_derivative, dtype=np.complex128)
    s_second = np.asarray(second_derivative, dtype=np.complex128)

    result = s_second + q1 * s_first + q0 * s_value

    if not np.all(np.isfinite(result)):
        raise ValueError("angular residual must be finite")

    return np.asarray(result, dtype=np.complex128)
