using Waves
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie
using Interpolations

struct AcousticDynamics <: AbstractDynamics
    c0::Float32
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor AcousticDynamics
Flux.trainable(::AcousticDynamics) = (;)

function AcousticDynamics(dim::AbstractDim, c0::Float32, pml_width::Float32, pml_scale::Float32)
    return AcousticDynamics(
        c0, 
        build_gradient(dim), 
        build_pml(dim, pml_width, pml_scale), 
        build_dirichlet(dim))
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

function (dyn::AcousticDynamics)(x, t::AbstractVector{Float32}, θ)

    C, F = θ

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

function Base.:∈(t::Float32, interp::DesignInterpolator)
    return interp.ti <= t <= interp.tf
end

function multi_design_interpolation(interps::Vector{DesignInterpolator}, t::Float32)
    _, idx = findmax(t .∈ interps)
    return interps[idx](t)
end

using ReinforcementLearning
include("../src/env.jl")

dim = TwoDim(15.0f0, 700)

n = 4

μ = zeros(Float32, n, 2)
μ[1, :] .= [-10.0f0, 0.0f0]
μ[2, :] .= [-10.0f0, 2.0f0]
μ[3, :] .= [-5.0f0, -5.0f0]
μ[4, :] .= [-5.0f0, 5.0f0]

σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.5f0
a[4] *= -1.0f0

pulse = build_normal(build_grid(dim), μ, σ, a)

F = Source(pulse, 1000.0f0)
env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = F,
    integration_steps = 1000
    ))

policy = RandomDesignPolicy(action_space(env))

a1 = policy(env)
a2 = policy(env)
a3 = policy(env)

@time tspan1, interp1, u_tot1 = cpu(env(a1))
@time tspan2, interp2, u_tot2 = cpu(env(a2))
@time tspan3, interp3, u_tot3 = cpu(env(a3))

tspan = flatten_repeated_last_dim(hcat(tspan1, tspan2, tspan3))
interp = [interp1, interp2, interp3]
u_tot = flatten_repeated_last_dim(cat(u_tot1, u_tot2, u_tot3, dims = 4))

println("Min: ", u_tot |> minimum)
println("Max: ", u_tot |> maximum)

u = linear_interpolation(tspan, Flux.unbatch(u_tot))

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)

@time record(fig, "vid.mp4", 1:5:length(tspan)) do i
    empty!(ax)
    heatmap!(ax, dim.x, dim.y, u(tspan[i]), colormap = :ice, colorrange = (-1.0f0, 1.0f0))
    mesh!(ax, multi_design_interpolation(interp, tspan[i]))
end