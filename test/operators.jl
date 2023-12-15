using Test
using Waves

@testset "Gradient" begin
    dim = OneDim(25.0f0, 1024)
    dx = get_dx(dim)
    grad = build_gradient(dim)

    ## y = x ^ 2
    y = dim.x .^ 2.0f0
    dydx_numerical = grad * y
    dydx_true = 2.0f0 * dim.x
    # @test all(dydx_true .≈ dydx_numerical)
    e = dydx_numerical .- dydx_true
    @test all(abs.(e) .< dx)
    
    ## y = sin(x)
    y = sin.(dim.x)
    dydx_numerical = grad * y
    dydx_true = cos.(dim.x)
    e = dydx_numerical .- dydx_true
    @test all(abs.(e) .< dx)

    ## y = exp(x)
    y = exp.(dim.x)
    dydx_numerical = grad * y
    dydx_true = exp.(dim.x)
    e = dydx_numerical .- dydx_true
    @test all(abs.(e) .< dx)
end

