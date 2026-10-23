#include "../cpp/high_order_time.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>

using namespace quassar::cfd;

int main() {
    try {
        const Gas gas{1.4};
        constexpr std::size_t n = 200;
        const double dx = 1.0 / static_cast<double>(n);

        Grid U;
        U.reserve(n);

        for (std::size_t i = 0; i < n; ++i) {
            const double x = (static_cast<double>(i) + 0.5) * dx;
            const double rho = (x >= 0.25 && x < 0.75) ? 1.0 : 0.5;

            U.push_back(
                primitive_to_conservative(rho, 0.2, 1.0, gas)
            );
        }

        const State initial = global_conserved_quantities(U, dx, gas);

        double time = 0.0;
        constexpr double final_time = 0.05;
        int steps = 0;

        while (time < final_time) {
            const double dt = std::min(
                stable_timestep(U, dx, 0.1, gas),
                final_time - time
            );

            if (!(dt > 0.0) || !std::isfinite(dt))
                throw std::runtime_error("Invalid time step");

            U = high_order_ssprk3_step(U, dt, dx, gas);
            time += dt;
            ++steps;
        }

        const State final = global_conserved_quantities(U, dx, gas);

        bool passed = true;
        double minimum_density = U.front()[0];
        double maximum_density = U.front()[0];
        double minimum_pressure = pressure(U.front(), gas);

        for (const State& cell : U) {
            minimum_density = std::min(minimum_density, cell[0]);
            maximum_density = std::max(maximum_density, cell[0]);
            minimum_pressure = std::min(
                minimum_pressure, pressure(cell, gas)
            );
        }

        for (std::size_t k = 0; k < 3; ++k) {
            const double error = std::abs(final[k] - initial[k]);
            const double tolerance =
                1e-10 * std::max(1.0, std::abs(initial[k]));

            std::cout << "Conservation component " << k
                      << ": error = " << error << '\n';

            if (!std::isfinite(error) || error > tolerance)
                passed = false;
        }

        std::cout << "Steps: " << steps << '\n';
        std::cout << "Minimum density: " << minimum_density << '\n';
        std::cout << "Maximum density: " << maximum_density << '\n';
        std::cout << "Minimum pressure: " << minimum_pressure << '\n';

        if (!(minimum_density > 0.0) ||
            !(minimum_pressure > 0.0))
            passed = false;

        if (!passed) {
            std::cerr << "RESEARCH 03: CONTACT ADVECTION TEST FAILED\n";
            return 1;
        }

        std::cout
            << "RESEARCH 03: CONTACT ADVECTION TEST PASSED\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RESEARCH 03: CONTACT ADVECTION TEST FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
