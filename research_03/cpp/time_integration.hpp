#pragma once

#include "finite_volume.hpp"

#include <cmath>
#include <stdexcept>

namespace quassar::cfd {

inline Grid linear_combination(
    const Grid& A,
    double alpha,
    const Grid& B,
    double beta
) {
    if (A.size() != B.size())
        throw std::invalid_argument("Grid size mismatch");

    Grid result(A.size());

    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            result[i][k] =
                alpha * A[i][k] + beta * B[i][k];
        }
    }

    return result;
}

inline Grid forward_euler_stage(
    const Grid& U,
    double dt,
    double dx,
    const Gas& gas
) {
    const Grid rhs = finite_volume_rhs(U, dx, gas);
    Grid result = U;

    for (std::size_t i = 0; i < U.size(); ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            result[i][k] += dt * rhs[i][k];
        }
        validate(result[i], gas);
    }

    return result;
}

inline double stable_timestep(
    const Grid& U,
    double dx,
    double cfl,
    const Gas& gas
) {
    if (U.empty() || !std::isfinite(dx) || dx <= 0.0 ||
        !std::isfinite(cfl) || cfl <= 0.0 || cfl > 1.0)
        throw std::invalid_argument("Invalid CFL parameters");

    double maximum_speed = 0.0;

    for (const State& cell : U) {
        const double speed =
            std::abs(velocity(cell, gas)) + sound_speed(cell, gas);

        maximum_speed = std::max(maximum_speed, speed);
    }

    if (!std::isfinite(maximum_speed) || maximum_speed <= 0.0)
        throw std::domain_error("Invalid characteristic speed");

    return cfl * dx / maximum_speed;
}

inline Grid ssprk3_step(
    const Grid& U,
    double dt,
    double dx,
    const Gas& gas
) {
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument("Invalid timestep");

    const Grid U1 = forward_euler_stage(U, dt, dx, gas);

    const Grid E1 = forward_euler_stage(U1, dt, dx, gas);
    const Grid U2 = linear_combination(U, 0.75, E1, 0.25);

    for (const State& cell : U2)
        validate(cell, gas);

    const Grid E2 = forward_euler_stage(U2, dt, dx, gas);
    const Grid U3 = linear_combination(U, 1.0 / 3.0, E2, 2.0 / 3.0);

    for (const State& cell : U3)
        validate(cell, gas);

    return U3;
}

} // namespace quassar::cfd
