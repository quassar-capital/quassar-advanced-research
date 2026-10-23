#include "../cpp/weno.hpp"

#include <array>
#include <cmath>
#include <iostream>

using namespace quassar::cfd;

int main() {
    constexpr int n = 200;
    constexpr double low = 0.125;
    constexpr double high = 1.0;
    constexpr double tolerance = 1e-8;

    double minimum = high;
    double maximum = low;

    for (int i = 2; i < n - 2; ++i) {
        std::array<double, 5> stencil{};

        for (int s = -2; s <= 2; ++s) {
            const int j = i + s;
            stencil[static_cast<std::size_t>(s + 2)] =
                j < n / 2 ? high : low;
        }

        const double value = weno_z5_left(stencil);

        if (!std::isfinite(value)) {
            std::cerr << "Nonfinite reconstruction\n";
            return 1;
        }

        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    std::cout << "Minimum reconstructed value: "
              << minimum << '\n';
    std::cout << "Maximum reconstructed value: "
              << maximum << '\n';

    if (minimum < low - tolerance ||
        maximum > high + tolerance) {
        std::cerr << "Discontinuity bounds test failed\n";
        return 1;
    }

    std::cout
        << "RESEARCH 03: WENO DISCONTINUITY TEST PASSED\n";
    return 0;
}
