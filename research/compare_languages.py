"""Independent cross-language numerical consistency experiment."""

import subprocess
import numpy as np

from quassar_advanced_research.transport import (
    TransportParameters,
    solve,
)


def parse_output(output):
    values = np.fromstring(output.strip(), sep="\n")
    if values.size != 128 or not np.all(np.isfinite(values)):
        raise RuntimeError("Invalid numerical output")
    return values


n = 128
x = (np.arange(n) + 0.5) / n
initial = 1 + 0.2 * np.sin(2 * np.pi * x)

python_result = solve(
    initial,
    TransportParameters(1.0, 0.02, 1.0),
    0.1,
)

rust_output = subprocess.check_output(
    ["cargo", "run", "--quiet", "--manifest-path",
     "rust/Cargo.toml", "--example", "compare"],
    text=True,
)
rust_result = parse_output(rust_output)

julia_output = subprocess.check_output(
    ["julia", "--project=julia", "research/compare_julia.jl"],
    text=True,
)
julia_result = parse_output(julia_output)

comparisons = {
    "Python vs Rust": np.max(np.abs(python_result - rust_result)),
    "Python vs Julia": np.max(np.abs(python_result - julia_result)),
    "Rust vs Julia": np.max(np.abs(rust_result - julia_result)),
}

print("CROSS-LANGUAGE NUMERICAL CONSISTENCY")
for name, error in comparisons.items():
    print(f"{name}: maximum absolute error = {error:.6e}")

assert all(error < 1e-12 for error in comparisons.values())
print("RESULT: PASS — all three implementations agree")
