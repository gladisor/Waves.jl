using Waves
using CairoMakie
using LinearAlgebra
using Flux

function linear_interpolation(X, Y, x)
    d = X .- x

    less_than = d .< 0.0f0
    greater_than = d .> 0.0f0
    less_than_or_equal_to = .!greater_than
    greater_than_or_equal_to = .!less_than
    equal_to = less_than_or_equal_to .& greater_than_or_equal_to

    if any(equal_to)
        return equal_to, equal_to
    else
        _, l = findmin(less_than) .- 1
        _, r = findmax(greater_than)
        return l, r
    end

    # if l == r
    #     return Y[l]
    # else
    #     display(l)
    #     display(r)
    #     dYdX = (Y[r] - Y[l]) / (X[r] - X[l])
    #     return Y[l] + dYdX * (x - X[l])
    # end
end

mask = I(10)

function polynomial_interpolation(X, Y, x)
    scale = maximum(abs.(X))
    n = ((X .- x) .* (.!mask) .+ mask) ./ scale
    d = (X .- X' .+ mask) ./ scale
    numer = dropdims(prod(n, dims = 1), dims = 1)
    denom = dropdims(prod(d, dims = 1), dims = 1)
    coef = numer ./ denom
    Y' * coef
end

gs = 5.0f0
dim = OneDim(gs, 512)
reduced_dim = OneDim(gs, 10)

x = reduced_dim.x
y = sin.(x) .* cos.(x) .+ x .^ 2

y_hat = [polynomial_interpolation(x, y, dim.x[i]) for i in axes(dim.x, 1)]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, dim.x, y_hat)
scatter!(ax, x, y)
save("interp.png", fig)
