using Waves
using Flux
using CairoMakie

struct AcousticDynamics{D <: AbstractDim} <: AbstractDynamics
    c0::Float32
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor AcousticDynamics
Flux.trainable(::AcousticDynamics) = (;)

function AcousticDynamics(dim::AbstractDim, c0::Float32, pml_width::Float32, pml_scale::Float32)
    
    return AcousticDynamics{typeof(dim)}(
        c0, 
        build_gradient(dim), 
        build_pml(dim, pml_width, pml_scale), 
        build_dirichlet(dim))
end

function (dyn::AcousticDynamics{OneDim})(x, t::AbstractVector{Float32}, θ)
    f = θ

    u = x[:, 1]
    v = x[:, 2]

    du = dyn.c0 .^ 2 * dyn.grad * v .- dyn.pml .* u
    dv = dyn.grad * (u .+ f(t)) .- dyn.pml .* v

    return hcat(du .* dyn.bc, dv)
end

function acoustic_dynamics(x, c, f, ∇, pml, bc)
    U = x[:, :, 1]
    Vx = x[:, :, 2]
    Vy = x[:, :, 3]
    Ψx = x[:, :, 4]
    Ψy = x[:, :, 5]
    Ω = x[:, :, 6]

    b = c .^ 2

    σx = pml
    σy = σx'

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U .+ f)
    Uy = ∂y(∇, U .+ f)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function (dyn::AcousticDynamics{TwoDim})(x, t::AbstractVector{Float32}, θ)
    c = C(t)
    f = F(t)

    dtot = acoustic_dynamics(x, c, f, dyn.grad, dyn.pml, dyn.bc)
    return dtot
end

function build_normal(x::AbstractVector{Float32}, μ::AbstractVector{Float32}, σ::AbstractVector, a::AbstractVector)

    μ = permutedims(μ)
    σ = permutedims(σ)
    a = permutedims(a)

    f = (1.0f0 ./ (σ * sqrt(2.0f0 * π))) .* a .* exp.(- ((x .- μ) .^ 2) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 2), dims = 2)
end

function build_normal(x::AbstractArray{Float32, 3}, μ::AbstractMatrix, σ::AbstractVector, a::AbstractVector)
    μ = permutedims(μ[:, :, :, :], (3, 4, 2, 1))
    σ = permutedims(σ[:, :, :], (2, 3, 1))
    a = permutedims(a[:, :, :], (2, 3, 1))
    
    f = (1.0f0 ./ (2.0f0 * π * σ .^ 2)) .* a .* exp.(-dropdims(sum((x .- μ) .^ 2, dims = 3), dims = 3) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 3), dims = 3)
end

dim = TwoDim(15.0f0, 256)
dyn = AcousticDynamics(dim, WATER, 2.0f0, 10000.0f0)
iter = Integrator(runge_kutta, dyn, 1f-5, 100)
tspan = build_tspan(iter, 0.0f0)
grid = build_grid(dim)

n = 1
μ = [-10.0f0, 0.0f0]'
σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 1.0f0

pulse = build_normal(grid, μ, σ, a)
F = Source(pulse, 1000.0f0)

design_space = Waves.build_triple_ring_design_space()
d1 = rand(design_space)
d2 = rand(design_space)
design_interpolator = DesignInterpolator(d1, d2, tspan[1], tspan[end])
C = t -> speed(design_interpolator(cpu(t[1])), grid, dyn.c0)

θ = [C, F]
x = build_wave(dim, 6)
@time sol = iter(x, tspan, θ)
println("Min: ", sol[:, :, 1, :] |> minimum)
println("Max: ", sol[:, :, 1, :] |> maximum)

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
@time record(fig, "vid.mp4", axes(sol, 4)) do i
    empty!(ax)
    heatmap!(ax, dim.x, dim.y, sol[:, :, 1, i], colormap = :ice, colorrange = (-1.0f0, 1.0f0))
end