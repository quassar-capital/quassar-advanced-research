#include "../cpp/high_order.hpp"

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
        const double x =
            (static_cast<double>(i) + 0.5) * dx;

        const double rho =
            1.0 + 0.15 * std::sin(2.0 * pi * x);

        const double u =
            0.2 + 0.05 * std::cos(2.0 * pi * x);

        const double p =
            1.0 + 0.1 * std::sin(4.0 * pi * x);

        U.push_back(
            primitive_to_conservative(rho, u, p, gas)
        );
    }

    const Grid rhs = high_order_rhs(U, dx, gas);

    State residual{0.0, 0.0, 0.0};
    State absolute_sum{0.0, 0.0, 0.0};

    for (const State& cell : rhs) {
        for (std::size_t k = 0; k < 3; ++k) {
            const double contribution = cell[k] * dx;

            residual[k] += contribution;
            absolute_sum[k] += std::abs(contribution);
        }
    }

    bool passed = true;

    for (std::size_t k = 0; k < 3; ++k) {
        const double tolerance =
            1e-12 * std::max(1.0, absolute_sum[k]);

        std::cout << "Component " << k
                  << ": conservation residual = "
                  << residual[k] << '\n';

        if (!std::isfinite(residual[k]) ||
            std::abs(residual[k]) > tolerance) {
            passed = false;
        }
    }

    if (!passed) {
        std::cerr
            << "HIGH-ORDER CONSERVATION TEST FAILED\n";
        return 1;
    }

    std::cout
        << "RESEARCH 03: HIGH-ORDER CONSERVATION PASSED\n";

    return 0;
}
