#pragma once

#include "finite_volume.hpp"
#include "weno.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quassar::cfd {

inline Grid high_order_rhs(
    const Grid& U,
    double dx,
    const Gas& gas
) {
    if (U.size() < 5)
        throw std::invalid_argument(
            "WENO-Z5 requires at least five cells"
        );

    if (!std::isfinite(dx) || dx <= 0.0)
        throw std::invalid_argument(
            "Invalid grid spacing"
        );

    const std::size_t n = U.size();

    for (const State& cell : U)
        validate(cell, gas);

    auto periodic = [n](std::ptrdiff_t j) -> std::size_t {
        const auto m = static_cast<std::ptrdiff_t>(n);
        return static_cast<std::size_t>((j % m + m) % m);
    };

    std::vector<State> fluxes(n);

    for (std::size_t i = 0; i < n; ++i) {
        State left{};
        State right{};

        const auto center = static_cast<std::ptrdiff_t>(i);

        for (std::size_t k = 0; k < 3; ++k) {
            std::array<double, 5> left_stencil{};
            std::array<double, 5> right_stencil{};

            for (int s = -2; s <= 2; ++s) {
                const std::size_t index =
                    static_cast<std::size_t>(s + 2);

                left_stencil[index] =
                    U[periodic(center + s)][k];

                right_stencil[index] =
                    U[periodic(center + 1 + s)][k];
            }

            left[k] = weno_z5_left(left_stencil);
            right[k] = weno_z5_right(right_stencil);
        }

        // Reject physically inadmissible reconstructed states.
        // Positivity-preserving limiting will be investigated
        // and verified separately.
        validate(left, gas);
        validate(right, gas);

        fluxes[i] = hll_flux(left, right, gas);
    }

    Grid rhs(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t previous = (i + n - 1) % n;

        for (std::size_t k = 0; k < 3; ++k) {
            rhs[i][k] =
                -(fluxes[i][k] - fluxes[previous][k]) / dx;
        }
    }

    return rhs;
}

} // namespace quassar::cfd
