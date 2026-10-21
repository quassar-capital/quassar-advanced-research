use quassar_multiphysics::{solve, TransportParameters};
use std::f64::consts::PI;

fn main() {
    let n = 128;
    let initial: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.2 * (2.0 * PI * (i as f64 + 0.5) / n as f64).sin())
        .collect();

    let p = TransportParameters {
        velocity: 1.0,
        diffusivity: 0.02,
        length: 1.0,
    };

    let result = solve(&initial, p, 0.1).unwrap();
    for value in result {
        println!("{value:.17e}");
    }
}
