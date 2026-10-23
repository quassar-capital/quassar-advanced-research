#include "../cpp/high_order_time.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

using namespace quassar::cfd;

int main() {
    const Gas gas{1.4};
    constexpr std::size_t n = 128;
    constexpr double pi = 3.14159265358979323846;
    const double dx = 1.0 / static_cast<double>(n);

    Grid U;
    U.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        const double x = (static_cast<double>(i) + 0.5) * dx;

        const double rho =
            1.0 + 0.1 * std::sin(2.0 * pi * x);

        U.push_back(
            primitive_to_conservative(rho, 0.3, 1.0, gas)
        );
    }

    const State initial =
        global_conserved_quantities(U, dx, gas);

    for (int step = 0; step < 50; ++step) {
        const double dt = stable_timestep(U, dx, 0.2, gas);
        U = high_order_ssprk3_step(U, dt, dx, gas);
    }

    const State final =
        global_conserved_quantities(U, dx, gas);

    bool passed = true;

    for (std::size_t k = 0; k < 3; ++k) {
        const double error =
            std::abs(final[k] - initial[k]);

        const double tolerance =
            1e-10 * std::max(1.0, std::abs(initial[k]));

        std::cout << "Component " << k
                  << ": conservation error = "
                  << error << '\n';

        if (!std::isfinite(error) || error > tolerance)
            passed = false;
    }

    if (!passed) {
        std::cerr << "HIGH-ORDER TIME TEST FAILED\n";
        return 1;
    }

    std::cout
        << "RESEARCH 03: HIGH-ORDER TIME VALIDATION PASSED\n";

    return 0;
}
