#include "../cpp/high_order.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace quassar::cfd;

int main() {
    const Gas gas{1.4};
    constexpr std::size_t n = 64;
    constexpr double dx = 1.0 / static_cast<double>(n);

    const State constant =
        primitive_to_conservative(1.0, 0.25, 1.0, gas);

    const Grid U(n, constant);
    const Grid rhs = high_order_rhs(U, dx, gas);

    double max_residual = 0.0;

    for (const State& cell : rhs) {
        for (double component : cell) {
            if (!std::isfinite(component)) {
                std::cerr << "Nonfinite residual\n";
                return 1;
            }
            max_residual =
                std::max(max_residual, std::abs(component));
        }
    }

    std::cout << "Maximum uniform-state residual: "
              << max_residual << '\n';

    if (max_residual > 1e-11) {
        std::cerr << "Uniform-state preservation failed\n";
        return 1;
    }

    std::cout
        << "RESEARCH 03: HIGH-ORDER OPERATOR BASIC TEST PASSED\n";

    return 0;
}
