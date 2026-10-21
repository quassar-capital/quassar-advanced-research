using Test
using QuassarMultiphysics

@testset "Quassar Multiphysics" begin
    n = 128
    x = ((collect(0:n-1) .+ 0.5) ./ n)
    initial = 1 .+ 0.2 .* sin.(2π .* x)

    @testset "Mass conservation" begin
        for velocity in (-2.0, 0.0, 2.0)
            for diffusivity in (0.0, 0.01, 0.1)
                p = TransportParameters(velocity, diffusivity)
                result = solve(initial, p, 0.1)
                @test abs(sum(result) - sum(initial)) / n < 1e-12
            end
        end
    end

    @testset "Constant states" begin
        p = TransportParameters(-3.0, 0.05)
        @test all(abs.(solve(fill(2.5, 64), p, 0.3) .- 2.5) .< 1e-12)
    end

    @testset "Conservative derivative" begin
        p = TransportParameters(1.7, 0.03)
        u = sin.(collect(1.0:97.0))
        @test abs(sum(rhs(u, p))) < 1e-10
    end

    @testset "Discrete maximum principle" begin
        p = TransportParameters(1.0, 0.02)
        u = [mod(i * 37, 101) / 25 - 1 for i in 1:80]
        result = solve(u, p, 0.2)
        @test minimum(result) >= minimum(u) - 1e-12
        @test maximum(result) <= maximum(u) + 1e-12
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError TransportParameters(1.0, -0.01)
        @test_throws ArgumentError TransportParameters(NaN, 0.01)
        @test_throws ArgumentError solve([1.0, 2.0], TransportParameters(1.0, 0.01), 0.1)
    end

    @testset "Stability" begin
        p = TransportParameters(2.0, 0.03)
        dt = stable_timestep(0.01, p)
        @test dt > 0
        @test dt * (2 / 0.01 + 2 * 0.03 / 0.01^2) <= 1
    end
end
