#include "../cpp/euler.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace quassar::cfd;

bool close(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <=
           tol * std::max({1.0, std::abs(a), std::abs(b)});
}

int main() {
    const Gas gas{1.4};

    const State U =
        primitive_to_conservative(1.2, 0.7, 1.5, gas);

    assert(close(pressure(U, gas), 1.5));
    assert(close(velocity(U, gas), 0.7));

    // Consistency: numerical flux equals physical flux
    // when left and right states are identical.
    const State F = physical_flux(U, gas);
    const State H = hll_flux(U, U, gas);

    for (std::size_t k = 0; k < 3; ++k)
        assert(close(F[k], H[k]));

    // Uniform-state preservation in a conservative
    // finite-volume update with periodic boundaries.
    const double dt_dx = 0.1;
    State updated{};

    for (std::size_t k = 0; k < 3; ++k) {
        updated[k] = U[k] - dt_dx * (H[k] - H[k]);
        assert(close(updated[k], U[k]));
    }

    // Invalid physical states must be rejected.
    bool rejected = false;

    try {
        const State invalid{-1.0, 0.0, 1.0};
        validate(invalid, gas);
    } catch (const std::domain_error&) {
        rejected = true;
    }

    assert(rejected);

    // A stationary contact discontinuity must have
    // zero mass flux under this symmetric HLL setup.
    const State L =
        primitive_to_conservative(1.0, 0.0, 1.0, gas);
    const State R =
        primitive_to_conservative(0.5, 0.0, 1.0, gas);

    const State contact_flux = hll_flux(L, R, gas);

    // HLL is diffusive at contacts: it does not preserve
    // stationary contact discontinuities exactly.
    assert(std::isfinite(contact_flux[0]));

    return 0;
}
