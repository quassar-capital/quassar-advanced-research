
"""Separated scalar Teukolsky operators on a subextremal Kerr background.

Conventions:
    Geometric units G = c = 1.
    Time dependence exp(-i omega t).
    Azimuthal dependence exp(i m phi).
    Scalar spin weight s = 0.

The angular separation constant is supplied explicitly. A complete
physical eigenvalue problem must determine it from the angular
boundary conditions and impose appropriate radial boundary conditions.

This module evaluates exact differential-equation coefficients and
residuals. It does not discretize the equations or compute Kerr modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class KerrBackground:
    mass: float
    spin: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("mass must be positive and finite")

        if not np.isfinite(self.spin):
            raise ValueError("spin must be finite")

        if abs(self.spin) >= self.mass:
            raise ValueError(
                "this formulation requires a subextremal Kerr background"
            )

    @property
    def dimensionless_spin(self) -> float:
        return self.spin / self.mass

    @property
    def extremality_gap(self) -> float:
        return float(
            np.sqrt(
                (1.0 - abs(self.dimensionless_spin))
                * (1.0 + abs(self.dimensionless_spin))
            )
        )

    @property
    def outer_horizon(self) -> float:
        return self.mass * (1.0 + self.extremality_gap)

    @property
    def inner_horizon(self) -> float:
        return self.mass * (1.0 - self.extremality_gap)

    @property
    def horizon_angular_velocity(self) -> float:
        return self.spin / (
            2.0 * self.mass * self.outer_horizon
        )

    def delta(self, radius: ArrayLike) -> NDArray[np.float64]:
        r = np.asarray(radius, dtype=np.float64)
        if not np.all(np.isfinite(r)):
            raise ValueError("radius must contain finite values")
        return (r - self.outer_horizon) * (
            r - self.inner_horizon
        )


@dataclass(frozen=True)
class ScalarMode:
    azimuthal_number: int
    frequency: complex
    angular_eigenvalue: complex

    def __post_init__(self) -> None:
        if (
            not isinstance(self.azimuthal_number, (int, np.integer))
            or isinstance(self.azimuthal_number, (bool, np.bool_))
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


def separation_constant(
    background: KerrBackground,
    mode: ScalarMode,
) -> complex:
    """Return the radial separation parameter lambda."""
    a = background.spin
    omega = complex(mode.frequency)
    m = mode.azimuthal_number

    return complex(
        mode.angular_eigenvalue
        + a * a * omega * omega
        - 2.0 * a * m * omega
    )


def radial_coefficients(
    background: KerrBackground,
    mode: ScalarMode,
    radius: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """Return coefficients p2, p1, p0 of p2 R'' + p1 R' + p0 R."""
    r = _exterior_radius(background, radius)

    delta = background.delta(r)
    delta_prime = 2.0 * (r - background.mass)

    omega = complex(mode.frequency)
    m = mode.azimuthal_number
    a = background.spin

    k = (r * r + a * a) * omega - a * m
    lam = separation_constant(background, mode)

    potential = k * k / delta - lam

    return (
        np.asarray(delta, dtype=np.float64),
        np.asarray(delta_prime, dtype=np.float64),
        np.asarray(potential, dtype=np.complex128),
    )


def angular_coefficients(
    background: KerrBackground,
    mode: ScalarMode,
    theta: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """Return coefficients q1, q0 of S'' + q1 S' + q0 S."""
    angle = _interior_angle(theta)

    sin_theta = np.sin(angle)
    cos_theta = np.cos(angle)

    omega = complex(mode.frequency)
    a = background.spin
    m = mode.azimuthal_number

    first_derivative = cos_theta / sin_theta

    potential = (
        (a * omega) ** 2 * cos_theta**2
        - m * m / sin_theta**2
        + complex(mode.angular_eigenvalue)
    )

    return (
        np.asarray(first_derivative, dtype=np.float64),
        np.asarray(potential, dtype=np.complex128),
    )


def radial_residual(
    background: KerrBackground,
    mode: ScalarMode,
    radius: ArrayLike,
    value: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> NDArray[np.complex128]:
    """Evaluate the scalar radial Teukolsky differential residual."""
    p2, p1, p0 = radial_coefficients(
        background, mode, radius
    )

    r_value = np.asarray(value, dtype=np.complex128)
    r_first = np.asarray(first_derivative, dtype=np.complex128)
    r_second = np.asarray(second_derivative, dtype=np.complex128)

    result = p2 * r_second + p1 * r_first + p0 * r_value

    if not np.all(np.isfinite(result)):
        raise ValueError("radial residual must be finite")

    return np.asarray(result, dtype=np.complex128)


def angular_residual(
    background: KerrBackground,
    mode: ScalarMode,
    theta: ArrayLike,
    value: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> NDArray[np.complex128]:
    """Evaluate the scalar angular Teukolsky differential residual."""
    q1, q0 = angular_coefficients(
        background, mode, theta
    )

    s_value = np.asarray(value, dtype=np.complex128)
    s_first = np.asarray(first_derivative, dtype=np.complex128)
    s_second = np.asarray(second_derivative, dtype=np.complex128)

    result = s_second + q1 * s_first + q0 * s_value

    if not np.all(np.isfinite(result)):
        raise ValueError("angular residual must be finite")

    return np.asarray(result, dtype=np.complex128)
