
#include "../cpp/high_order.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace quassar::cfd;

constexpr double pi = 3.14159265358979323846;

double density_average(double x, double dx) {
    const double factor = std::sin(pi * dx) / (pi * dx);
    return 1.0 + 0.2 * factor * std::sin(2.0 * pi * x);
}

double convergence_error(int n) {
    const Gas gas{1.4};
    const double dx = 1.0 / static_cast<double>(n);
    constexpr double u = 0.5;
    constexpr double p = 1.0;

    Grid U;
    U.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) * dx;
        U.push_back(primitive_to_conservative(
            density_average(x, dx), u, p, gas
        ));
    }

    const Grid rhs = high_order_rhs(U, dx, gas);
    double error = 0.0;

    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) * dx;
        const double exact =
    -u * 0.4 * pi
    * (std::sin(pi * dx) / (pi * dx))
    * std::cos(2.0 * pi * x);

        error += std::abs(rhs[i][0] - exact);
    }

    return error / static_cast<double>(n);
}

int main() {
    const std::vector<int> sizes{20, 40, 80, 160, 320};
    double previous = 0.0;

    std::cout << std::setprecision(12);
    std::cout << "Cells | Density RHS L1 error | Order\n";

    for (int n : sizes) {
        const double error = convergence_error(n);

        if (!std::isfinite(error) || error <= 0.0) {
            std::cerr << "Invalid convergence error\n";
            return 1;
        }

        const double order =
            previous > 0.0
                ? std::log2(previous / error)
                : 0.0;

        std::cout << n << " | " << error
                  << " | " << order << '\n';

        previous = error;
    }

    std::cout
        << "RESEARCH 03: EULER CONVERGENCE DATA GENERATED\n";

    return 0;
}
