#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace quassar::cfd {

using State = std::array<double, 3>;

struct Gas {
    double gamma = 1.4;

    explicit Gas(double g = 1.4) : gamma(g) {
        if (!std::isfinite(g) || g <= 1.0)
            throw std::invalid_argument("Invalid heat capacity ratio");
    }
};

inline void validate(const State& U, const Gas& gas) {
    for (double x : U)
        if (!std::isfinite(x))
            throw std::domain_error("Nonfinite conservative state");

    if (U[0] <= 0.0)
        throw std::domain_error("Nonpositive density");

    const double internal = U[2] - 0.5 * U[1] * (U[1] / U[0]);

    if (!std::isfinite(internal) || internal <= 0.0)
        throw std::domain_error("Nonpositive internal energy");

    const double p = (gas.gamma - 1.0) * internal;

    if (!std::isfinite(p) || p <= 0.0)
        throw std::domain_error("Nonpositive pressure");
}

inline double pressure(const State& U, const Gas& gas) {
    validate(U, gas);
    return (gas.gamma - 1.0) *
           (U[2] - 0.5 * U[1] * (U[1] / U[0]));
}

inline double velocity(const State& U, const Gas& gas) {
    validate(U, gas);
    return U[1] / U[0];
}

inline double sound_speed(const State& U, const Gas& gas) {
    return std::sqrt(gas.gamma * pressure(U, gas) / U[0]);
}

inline State physical_flux(const State& U, const Gas& gas) {
    const double u = velocity(U, gas);
    const double p = pressure(U, gas);

    return {
        U[1],
        U[1] * u + p,
        u * (U[2] + p)
    };
}

inline State primitive_to_conservative(
    double rho, double u, double p, const Gas& gas
) {
    if (!std::isfinite(rho) || !std::isfinite(u) ||
        !std::isfinite(p) || rho <= 0.0 || p <= 0.0)
        throw std::domain_error("Invalid primitive state");

    State U{
        rho,
        rho * u,
        p / (gas.gamma - 1.0) + 0.5 * rho * u * u
    };

    validate(U, gas);
    return U;
}

inline State hll_flux(
    const State& L, const State& R, const Gas& gas
) {
    const double uL = velocity(L, gas);
    const double uR = velocity(R, gas);
    const double aL = sound_speed(L, gas);
    const double aR = sound_speed(R, gas);

    const double sL = std::min(uL - aL, uR - aR);
    const double sR = std::max(uL + aL, uR + aR);

    const State FL = physical_flux(L, gas);
    const State FR = physical_flux(R, gas);

    if (sL >= 0.0) return FL;
    if (sR <= 0.0) return FR;

    if (!(sR > sL))
        throw std::domain_error("Degenerate HLL wave speeds");

    State F{};

    for (std::size_t k = 0; k < 3; ++k) {
        F[k] = (sR * FL[k] - sL * FR[k]
              + sL * sR * (R[k] - L[k])) / (sR - sL);
    }

    return F;
}

} // namespace quassar::cfd
