//! Conservative finite-volume advection-diffusion solver.
//! Periodic boundary conditions and explicit Euler integration.

#[derive(Debug, Clone, Copy)]
pub struct TransportParameters {
    pub velocity: f64,
    pub diffusivity: f64,
    pub length: f64,
}

impl TransportParameters {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.velocity.is_finite() {
            return Err("velocity must be finite");
        }
        if !self.diffusivity.is_finite() || self.diffusivity < 0.0 {
            return Err("diffusivity must be finite and nonnegative");
        }
        if !self.length.is_finite() || self.length <= 0.0 {
            return Err("length must be finite and positive");
        }
        Ok(())
    }
}

pub fn stable_timestep(dx: f64, p: TransportParameters, safety: f64) -> Result<f64, &'static str> {
    p.validate()?;
    if !dx.is_finite() || dx <= 0.0 {
        return Err("dx must be positive and finite");
    }
    if !safety.is_finite() || safety <= 0.0 || safety > 1.0 {
        return Err("safety must be in (0, 1]");
    }
    let rate = p.velocity.abs() / dx + 2.0 * p.diffusivity / (dx * dx);
    if rate == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(safety / rate)
    }
}

pub fn rhs(u: &[f64], p: TransportParameters) -> Result<Vec<f64>, &'static str> {
    p.validate()?;
    if u.len() < 3 || u.iter().any(|x| !x.is_finite()) {
        return Err("state must contain at least three finite cells");
    }

    let n = u.len();
    let dx = p.length / n as f64;
    let mut flux = vec![0.0; n];

    for i in 0..n {
        let right = u[(i + 1) % n];
        let upwind = if p.velocity >= 0.0 { u[i] } else { right };
        flux[i] = p.velocity * upwind - p.diffusivity * (right - u[i]) / dx;
    }

    let mut derivative = vec![0.0; n];
    for i in 0..n {
        let left = (i + n - 1) % n;
        derivative[i] = -(flux[i] - flux[left]) / dx;
    }
    Ok(derivative)
}

pub fn solve(
    initial: &[f64],
    p: TransportParameters,
    final_time: f64,
) -> Result<Vec<f64>, &'static str> {
    if !final_time.is_finite() || final_time < 0.0 {
        return Err("final_time must be finite and nonnegative");
    }
    p.validate()?;
    if initial.len() < 3 || initial.iter().any(|x| !x.is_finite()) {
        return Err("invalid initial state");
    }

    let mut u = initial.to_vec();
    if final_time == 0.0 {
        return Ok(u);
    }

    let dx = p.length / u.len() as f64;
    let dt_limit = stable_timestep(dx, p, 0.8)?;
    let steps = ((final_time / dt_limit).ceil() as usize).max(1);
    let dt = final_time / steps as f64;

    for _ in 0..steps {
        let du = rhs(&u, p)?;
        for (value, derivative) in u.iter_mut().zip(du.iter()) {
            *value += dt * derivative;
        }
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn parameters(velocity: f64, diffusivity: f64) -> TransportParameters {
        TransportParameters {
            velocity,
            diffusivity,
            length: 1.0,
        }
    }

    #[test]
    fn conserves_mass_across_regimes() {
        let n = 128;
        let u: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.2 * (2.0 * PI * (i as f64 + 0.5) / n as f64).sin())
            .collect();

        for velocity in [-2.0, 0.0, 2.0] {
            for diffusivity in [0.0, 0.01, 0.1] {
                let v = solve(&u, parameters(velocity, diffusivity), 0.1).unwrap();
                let error = (u.iter().sum::<f64>() - v.iter().sum::<f64>()).abs() / n as f64;
                assert!(error < 1e-12, "mass error: {error}");
            }
        }
    }

    #[test]
    fn preserves_constant_states() {
        let u = vec![2.5; 64];
        let v = solve(&u, parameters(-3.0, 0.05), 0.3).unwrap();
        assert!(v.iter().all(|x| (x - 2.5).abs() < 1e-12));
    }

    #[test]
    fn satisfies_discrete_maximum_principle() {
        let u: Vec<f64> = (0..80)
            .map(|i| ((i * 37 % 101) as f64) / 25.0 - 1.0)
            .collect();
        let v = solve(&u, parameters(1.0, 0.02), 0.2).unwrap();

        let min = u.iter().copied().fold(f64::INFINITY, f64::min);
        let max = u.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        assert!(v.iter().all(|x| *x >= min - 1e-12 && *x <= max + 1e-12));
    }

    #[test]
    fn rejects_invalid_physics() {
        assert!(parameters(1.0, -0.01).validate().is_err());
        assert!(parameters(f64::NAN, 0.01).validate().is_err());
        assert!(solve(&[1.0, 2.0], parameters(1.0, 0.01), 0.1).is_err());
    }

    #[test]
    fn conservative_rhs_sums_to_zero() {
        let u: Vec<f64> = (0..97).map(|i| (i as f64).sin()).collect();
        let du = rhs(&u, parameters(1.7, 0.03)).unwrap();
        assert!(du.iter().sum::<f64>().abs() < 1e-10);
    }
}
