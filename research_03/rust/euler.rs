#[derive(Clone, Copy, Debug)]
pub struct Gas {
    pub gamma: f64,
}

pub type State = [f64; 3];

impl Gas {
    pub fn new(gamma: f64) -> Result<Self, &'static str> {
        if !gamma.is_finite() || gamma <= 1.0 {
            return Err("Invalid heat-capacity ratio");
        }
        Ok(Self { gamma })
    }
}

pub fn conservative(
    rho: f64,
    velocity: f64,
    pressure: f64,
    gas: Gas,
) -> Result<State, &'static str> {
    if !rho.is_finite()
        || !velocity.is_finite()
        || !pressure.is_finite()
        || rho <= 0.0
        || pressure <= 0.0
    {
        return Err("Invalid primitive state");
    }

    let state = [
        rho,
        rho * velocity,
        pressure / (gas.gamma - 1.0)
            + 0.5 * rho * velocity * velocity,
    ];

    validate(state, gas)?;
    Ok(state)
}

pub fn validate(
    state: State,
    gas: Gas,
) -> Result<(), &'static str> {
    if state.iter().any(|x| !x.is_finite()) || state[0] <= 0.0 {
        return Err("Invalid conservative state");
    }

    let pressure = (gas.gamma - 1.0)
        * (state[2] - 0.5 * state[1] * state[1] / state[0]);

    if !pressure.is_finite() || pressure <= 0.0 {
        return Err("Nonpositive pressure");
    }

    Ok(())
}

pub fn pressure(
    state: State,
    gas: Gas,
) -> Result<f64, &'static str> {
    validate(state, gas)?;
    Ok((gas.gamma - 1.0)
        * (state[2] - 0.5 * state[1] * state[1] / state[0]))
}

pub fn physical_flux(
    state: State,
    gas: Gas,
) -> Result<State, &'static str> {
    let p = pressure(state, gas)?;
    let velocity = state[1] / state[0];

    let flux = [
        state[1],
        state[1] * velocity + p,
        velocity * (state[2] + p),
    ];

    if flux.iter().any(|x| !x.is_finite()) {
        return Err("Nonfinite physical flux");
    }

    Ok(flux)
}

pub fn hll_flux(
    left: State,
    right: State,
    gas: Gas,
) -> Result<State, &'static str> {
    let p_l = pressure(left, gas)?;
    let p_r = pressure(right, gas)?;

    let u_l = left[1] / left[0];
    let u_r = right[1] / right[0];

    let a_l = (gas.gamma * p_l / left[0]).sqrt();
    let a_r = (gas.gamma * p_r / right[0]).sqrt();

    let s_l = (u_l - a_l).min(u_r - a_r);
    let s_r = (u_l + a_l).max(u_r + a_r);

    if !s_l.is_finite() || !s_r.is_finite() {
        return Err("Nonfinite HLL wave speed");
    }

    let f_l = physical_flux(left, gas)?;
    let f_r = physical_flux(right, gas)?;

    if s_l >= 0.0 {
        return Ok(f_l);
    }

    if s_r <= 0.0 {
        return Ok(f_r);
    }

    let mut result = [0.0; 3];

    for k in 0..3 {
        result[k] = (
            s_r * f_l[k]
                - s_l * f_r[k]
                + s_l * s_r * (right[k] - left[k])
        ) / (s_r - s_l);
    }

    if result.iter().any(|x| !x.is_finite()) {
        return Err("Nonfinite HLL flux");
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-11, "{a} != {b}");
    }

    #[test]
    fn conservative_state_and_physical_flux() {
        let gas = Gas::new(1.4).unwrap();
        let u = conservative(1.0, 2.0, 1.0, gas).unwrap();

        for (a, b) in u.iter().zip([1.0, 2.0, 4.5]) {
            close(*a, b);
        }

        let f = physical_flux(u, gas).unwrap();

        for (a, b) in f.iter().zip([2.0, 5.0, 11.0]) {
            close(*a, b);
        }
    }

    #[test]
    fn hll_consistency() {
        let gas = Gas::new(1.4).unwrap();
        let u = conservative(1.0, 2.0, 1.0, gas).unwrap();

        let numerical = hll_flux(u, u, gas).unwrap();
        let physical = physical_flux(u, gas).unwrap();

        for k in 0..3 {
            close(numerical[k], physical[k]);
        }
    }

    #[test]
    fn interface_conservation() {
        let gas = Gas::new(1.4).unwrap();

        let left = conservative(1.0, 0.0, 1.0, gas).unwrap();
        let right = conservative(0.125, 0.0, 0.1, gas).unwrap();

        let flux = hll_flux(left, right, gas).unwrap();
        let ratio = 0.01;

        for k in 0..3 {
            let initial = left[k] + right[k];
            let updated =
                (left[k] - ratio * flux[k])
                + (right[k] + ratio * flux[k]);

            close(initial, updated);
        }
    }

    #[test]
    fn invalid_states_rejected() {
        let gas = Gas::new(1.4).unwrap();

        assert!(Gas::new(1.0).is_err());
        assert!(Gas::new(f64::NAN).is_err());

        for (rho, velocity, p) in [
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, -1.0),
            (f64::NAN, 0.0, 1.0),
            (1.0, f64::INFINITY, 1.0),
        ] {
            assert!(conservative(rho, velocity, p, gas).is_err());
        }
    }
}
