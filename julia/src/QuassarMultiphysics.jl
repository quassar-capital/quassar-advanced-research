module QuassarMultiphysics

export TransportParameters, stable_timestep, rhs, solve

struct TransportParameters
    velocity::Float64
    diffusivity::Float64
    length::Float64

    function TransportParameters(velocity, diffusivity, length=1.0)
        a, k, l = Float64(velocity), Float64(diffusivity), Float64(length)
        isfinite(a) || throw(ArgumentError("velocity must be finite"))
        isfinite(k) && k >= 0 ||
            throw(ArgumentError("diffusivity must be finite and nonnegative"))
        isfinite(l) && l > 0 ||
            throw(ArgumentError("length must be finite and positive"))
        new(a, k, l)
    end
end

function stable_timestep(dx, p::TransportParameters; safety=0.8)
    isfinite(dx) && dx > 0 ||
        throw(ArgumentError("dx must be finite and positive"))
    isfinite(safety) && 0 < safety <= 1 ||
        throw(ArgumentError("safety must lie in (0, 1]"))

    rate = abs(p.velocity) / dx + 2p.diffusivity / dx^2
    rate == 0 ? Inf : safety / rate
end

function rhs(u::AbstractVector, p::TransportParameters)
    n = length(u)
    n >= 3 || throw(ArgumentError("at least three cells required"))
    all(isfinite, u) || throw(ArgumentError("state must be finite"))

    dx = p.length / n
    flux = Vector{Float64}(undef, n)

    for i in 1:n
        right = u[mod1(i + 1, n)]
        upwind = p.velocity >= 0 ? u[i] : right
        flux[i] = p.velocity * upwind -
                  p.diffusivity * (right - u[i]) / dx
    end

    derivative = Vector{Float64}(undef, n)
    for i in 1:n
        derivative[i] = -(flux[i] - flux[mod1(i - 1, n)]) / dx
    end

    derivative
end

function solve(u0::AbstractVector, p::TransportParameters, final_time;
               safety=0.8)
    isfinite(final_time) && final_time >= 0 ||
        throw(ArgumentError("final_time must be finite and nonnegative"))

    u = Float64.(u0)
    length(u) >= 3 || throw(ArgumentError("at least three cells required"))
    all(isfinite, u) || throw(ArgumentError("state must be finite"))

    final_time == 0 && return u

    dx = p.length / length(u)
    dt_limit = stable_timestep(dx, p; safety=safety)
    steps = max(1, ceil(Int, final_time / dt_limit))
    dt = final_time / steps

    for _ in 1:steps
        u .+= dt .* rhs(u, p)
    end

    u
end

end
