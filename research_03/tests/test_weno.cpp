#include "../cpp/weno.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace quassar::cfd;

bool close(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <=
           tol * std::max({1.0, std::abs(a), std::abs(b)});
}

int main() {
    // Constant-state reproduction.
    const std::array<double, 5> constant{
        2.5, 2.5, 2.5, 2.5, 2.5
    };

    if (!close(weno_z5_left(constant), 2.5) ||
        !close(weno_z5_right(constant), 2.5)) {
        std::cerr << "Constant-state reconstruction failed\n";
        return 1;
    }

    // Linear-profile reconstruction at the two interfaces
    // surrounding the central cell.
    const std::array<double, 5> linear{
        -2.0, -1.0, 0.0, 1.0, 2.0
    };

    if (!close(weno_z5_left(linear), 0.5) ||
        !close(weno_z5_right(linear), -0.5)) {
        std::cerr << "Linear-profile reconstruction failed\n";
        return 1;
    }

    // Finite reconstruction near a discontinuity.
    const std::array<double, 5> discontinuity{
        1.0, 1.0, 1.0, 0.125, 0.125
    };

    if (!std::isfinite(weno_z5_left(discontinuity)) ||
        !std::isfinite(weno_z5_right(discontinuity))) {
        std::cerr << "Discontinuity reconstruction failed\n";
        return 1;
    }

    // Nonfinite input must be rejected.
    auto invalid = constant;
    invalid[2] = std::numeric_limits<double>::quiet_NaN();

    bool rejected = false;

    try {
        (void)weno_z5_left(invalid);
    } catch (const std::domain_error&) {
        rejected = true;
    }

    if (!rejected) {
        std::cerr << "Invalid-input rejection failed\n";
        return 1;
    }

    std::cout << "RESEARCH 03: WENO-Z5 BASIC TESTS PASSED\n";
    return 0;
}
