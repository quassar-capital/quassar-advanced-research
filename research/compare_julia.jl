using QuassarMultiphysics

n = 128
x = (collect(0:n-1) .+ 0.5) ./ n
initial = 1 .+ 0.2 .* sin.(2π .* x)
p = TransportParameters(1.0, 0.02, 1.0)

result = solve(initial, p, 0.1)

for value in result
    println(value)
end
