#pragma once

#include "euler.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace quassar::cfd {

using Grid = std::vector<State>;

inline Grid finite_volume_rhs(
    const Grid& U,
    double dx,
    const Gas& gas
) {
    if (U.size() < 3)
        throw std::invalid_argument(
            "At least three finite-volume cells required"
        );

    if (!std::isfinite(dx) || dx <= 0.0)
        throw std::invalid_argument(
            "Invalid grid spacing"
        );

    const std::size_t n = U.size();

    for (const State& cell : U)
        validate(cell, gas);

    // One numerical flux per periodic cell interface.
    std::vector<State> interface_flux(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t next = (i + 1) % n;

        interface_flux[i] =
            hll_flux(U[i], U[next], gas);
    }

    Grid rhs(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t previous = (i + n - 1) % n;

        for (std::size_t k = 0; k < 3; ++k) {
            rhs[i][k] =
                -(interface_flux[i][k]
                - interface_flux[previous][k]) / dx;
        }
    }

    return rhs;
}

inline State global_conserved_quantities(
    const Grid& U,
    double dx,
    const Gas& gas
) {
    if (U.empty() || !std::isfinite(dx) || dx <= 0.0)
        throw std::invalid_argument("Invalid grid");

    State totals{0.0, 0.0, 0.0};

    for (const State& cell : U) {
        validate(cell, gas);

        for (std::size_t k = 0; k < 3; ++k)
            totals[k] += cell[k] * dx;
    }

    return totals;
}

} // namespace quassar::cfd
