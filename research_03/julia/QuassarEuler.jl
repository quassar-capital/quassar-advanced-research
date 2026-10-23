module QuassarEuler

export Gas, conservative_state, pressure, physical_flux, hll_flux

struct Gas
    gamma::Float64

    function Gas(gamma::Real)
        g = Float64(gamma)
        isfinite(g) && g > 1.0 ||
            throw(ArgumentError("gamma must be finite and greater than one"))
        new(g)
    end
end

function conservative_state(
    rho::Real,
    velocity::Real,
    p::Real,
    gas::Gas
)
    r = Float64(rho)
    v = Float64(velocity)
    pressure_value = Float64(p)

    all(isfinite, (r, v, pressure_value)) ||
        throw(ArgumentError("Nonfinite primitive state"))

    r > 0.0 && pressure_value > 0.0 ||
        throw(ArgumentError("Density and pressure must be positive"))

    energy = pressure_value / (gas.gamma - 1.0) +
             0.5 * r * v^2

    U = (r, r * v, energy)
    validate(U, gas)
    return U
end

function validate(U, gas::Gas)
    all(isfinite, U) ||
        throw(ArgumentError("Nonfinite conservative state"))

    rho, momentum, energy = U

    rho > 0.0 ||
        throw(ArgumentError("Nonpositive density"))

    p = (gas.gamma - 1.0) *
        (energy - 0.5 * momentum^2 / rho)

    isfinite(p) && p > 0.0 ||
        throw(ArgumentError("Nonpositive pressure"))

    return nothing
end

function pressure(U, gas::Gas)
    validate(U, gas)
    rho, momentum, energy = U
    return (gas.gamma - 1.0) *
           (energy - 0.5 * momentum^2 / rho)
end

function physical_flux(U, gas::Gas)
    validate(U, gas)

    rho, momentum, energy = U
    velocity = momentum / rho
    p = pressure(U, gas)

    return (
        momentum,
        momentum * velocity + p,
        velocity * (energy + p)
    )
end

function hll_flux(left, right, gas::Gas)
    validate(left, gas)
    validate(right, gas)

    rho_l, momentum_l, _ = left
    rho_r, momentum_r, _ = right

    velocity_l = momentum_l / rho_l
    velocity_r = momentum_r / rho_r

    pressure_l = pressure(left, gas)
    pressure_r = pressure(right, gas)

    sound_l = sqrt(gas.gamma * pressure_l / rho_l)
    sound_r = sqrt(gas.gamma * pressure_r / rho_r)

    s_l = min(velocity_l - sound_l, velocity_r - sound_r)
    s_r = max(velocity_l + sound_l, velocity_r + sound_r)

    flux_l = physical_flux(left, gas)
    flux_r = physical_flux(right, gas)

    if s_l >= 0.0
        return flux_l
    elseif s_r <= 0.0
        return flux_r
    end

    return ntuple(
        i -> (
            s_r * flux_l[i] -
            s_l * flux_r[i] +
            s_l * s_r * (right[i] - left[i])
        ) / (s_r - s_l),
        3
    )
end

end
