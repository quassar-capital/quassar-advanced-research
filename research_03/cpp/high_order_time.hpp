#pragma once

#include "high_order.hpp"
#include "time_integration.hpp"

#include <cmath>
#include <stdexcept>

namespace quassar::cfd {

inline Grid high_order_ssprk3_step(
    const Grid& U,
    double dt,
    double dx,
    const Gas& gas
) {
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument("Invalid time step");

    const Grid k1 = high_order_rhs(U, dx, gas);

    Grid U1 = U;
    for (std::size_t i = 0; i < U.size(); ++i)
        for (std::size_t k = 0; k < 3; ++k)
            U1[i][k] += dt * k1[i][k];

    const Grid k2 = high_order_rhs(U1, dx, gas);

    Grid U2 = U;
    for (std::size_t i = 0; i < U.size(); ++i)
        for (std::size_t k = 0; k < 3; ++k)
            U2[i][k] = 0.75 * U[i][k]
                + 0.25 * (U1[i][k] + dt * k2[i][k]);

    const Grid k3 = high_order_rhs(U2, dx, gas);

    Grid result = U;
    for (std::size_t i = 0; i < U.size(); ++i)
        for (std::size_t k = 0; k < 3; ++k)
            result[i][k] = (1.0 / 3.0) * U[i][k]
                + (2.0 / 3.0) * (U2[i][k] + dt * k3[i][k]);

    for (const State& cell : result)
        validate(cell, gas);

    return result;
}

} // namespace quassar::cfd
