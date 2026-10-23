#include "../cpp/weno.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace quassar::cfd;

constexpr double pi = 3.14159265358979323846;

double exact_interface(double x) {
    return 1.0 + 0.2 * std::sin(2.0 * pi * x);
}

double cell_average(int j, int n) {
    const double dx = 1.0 / static_cast<double>(n);
    const double x = (static_cast<double>(j) + 0.5) * dx;
    const double factor = std::sin(pi * dx) / (pi * dx);

    return 1.0 + 0.2 * factor * std::sin(2.0 * pi * x);
}

double reconstruction_error(int n) {
    const double dx = 1.0 / static_cast<double>(n);
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        std::array<double, 5> stencil{};

        for (int s = -2; s <= 2; ++s) {
            const int j = (i + s + n) % n;
            stencil[static_cast<std::size_t>(s + 2)] =
                cell_average(j, n);
        }

        const double reconstructed = weno_z5_left(stencil);
        const double exact = exact_interface((i + 1.0) * dx);

        sum += std::abs(reconstructed - exact);
    }

    return sum / static_cast<double>(n);
}

int main() {
    const std::vector<int> resolutions{20, 40, 80, 160, 320};
    double previous_error = 0.0;
    double last_order = 0.0;

    std::cout << std::setprecision(12);
    std::cout << "Cells | L1 error | Observed order\n";

    for (int n : resolutions) {
        const double error = reconstruction_error(n);

        if (!std::isfinite(error) || error <= 0.0) {
            std::cerr << "Invalid reconstruction error\n";
            return 1;
        }

        double order = 0.0;

        if (previous_error > 0.0) {
            order = std::log2(previous_error / error);
            last_order = order;
        }

        std::cout << n << " | " << error
                  << " | " << order << '\n';

        previous_error = error;
    }

    if (last_order < 4.5) {
        std::cerr << "WENO-Z5 convergence requirement not met\n";
        return 1;
    }

    std::cout << "RESEARCH 03: WENO-Z5 CONVERGENCE PASSED\n";
    return 0;
}
