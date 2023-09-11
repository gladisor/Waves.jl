using Waves
using CairoMakie
using LinearAlgebra
using Flux

mask = I(10)
function polynomial_interpolation(X, Y, x)
    scale = maximum(abs.(X))
    n = ((X .- x) .* (.!mask) .+ mask) ./ scale
    d = (X .- X' .+ mask) ./ scale
    numer = dropdims(prod(n, dims = 1), dims = 1)
    denom = dropdims(prod(d, dims = 1), dims = 1)
    coef = numer ./ denom
    return Y' * coef
end

grid_size = 5.0f0
dim = OneDim(grid_size, 512)
reduced_dim = OneDim(grid_size, 10)

x = reduced_dim.x
y = sin.(x) .* cos.(x) .+ x .^ 2

y_hat = [polynomial_interpolation(x, y, dim.x[i]) for i in axes(dim.x, 1)]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, dim.x, y_hat)
scatter!(ax, x, y)
save("interp.png", fig)
