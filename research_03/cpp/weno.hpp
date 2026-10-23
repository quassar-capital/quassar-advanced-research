#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace quassar::cfd {

inline double weno_z5_left(
    const std::array<double, 5>& v
) {
    for (double x : v) {
        if (!std::isfinite(x))
            throw std::domain_error("Nonfinite WENO stencil");
    }

    const double b0 =
        (13.0 / 12.0) *
        std::pow(v[0] - 2.0*v[1] + v[2], 2) +
        0.25 *
        std::pow(v[0] - 4.0*v[1] + 3.0*v[2], 2);

    const double b1 =
        (13.0 / 12.0) *
        std::pow(v[1] - 2.0*v[2] + v[3], 2) +
        0.25 *
        std::pow(v[1] - v[3], 2);

    const double b2 =
        (13.0 / 12.0) *
        std::pow(v[2] - 2.0*v[3] + v[4], 2) +
        0.25 *
        std::pow(3.0*v[2] - 4.0*v[3] + v[4], 2);

    const double q0 =
        (2.0*v[0] - 7.0*v[1] + 11.0*v[2]) / 6.0;

    const double q1 =
        (-v[1] + 5.0*v[2] + 2.0*v[3]) / 6.0;

    const double q2 =
        (2.0*v[2] + 5.0*v[3] - v[4]) / 6.0;

    const double tau5 = std::abs(b0 - b2);

    constexpr double eps = 1e-12;

    const double a0 =
        0.1 * (1.0 + std::pow(tau5 / (b0 + eps), 2));

    const double a1 =
        0.6 * (1.0 + std::pow(tau5 / (b1 + eps), 2));

    const double a2 =
        0.3 * (1.0 + std::pow(tau5 / (b2 + eps), 2));

    const double sum = a0 + a1 + a2;

    if (!std::isfinite(sum) || sum <= 0.0)
        throw std::domain_error("Invalid WENO weights");

    return (a0*q0 + a1*q1 + a2*q2) / sum;
}

inline double weno_z5_right(
    const std::array<double, 5>& v
) {
    return weno_z5_left({
        v[4], v[3], v[2], v[1], v[0]
    });
}

} // namespace quassar::cfd
