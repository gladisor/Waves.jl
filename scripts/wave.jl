using Waves
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie
using Interpolations
import ProgressMeter
using ReinforcementLearning
using Images: imresize
include("../src/env.jl")

const FRAMES_PER_SECOND = 24

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

function render!(policy::AbstractPolicy, env::WaveEnv, seconds::Float32)
    tspans = []
    interps = DesignInterpolator[]
    u_tots = []

    # p = ProgressMeter.Progress(n, "Simulating", env.actions)
    reset!(env)
    while !is_terminated(env)
        tspan, interp, u_tot = cpu(env(policy(env)))
        push!(tspans, tspan)
        push!(interps, interp)
        push!(u_tots, u_tot)
        println(env.time_step)
        # ProgressMeter.next!(p)
    end
    # ProgressMeter.finish!(p)

    tspan = flatten_repeated_last_dim(hcat(tspans...))
    u_tot = flatten_repeated_last_dim(cat(u_tots..., dims = 4))
    u_tot = linear_interpolation(tspan, Flux.unbatch(u_tot))

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    # p = ProgressMeter.Progress(n, "Rendering", length(tspan))
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)
    @time record(fig, "vid.mp4", tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax)
        heatmap!(ax, dim.x, dim.y, u_tot(t), colormap = :ice, colorrange = (-1.0f0, 1.0f0))
        mesh!(ax, multi_design_interpolation(interps, t))
        # ProgressMeter.next!(p)
    end
    # ProgressMeter.finish!(p)

    return nothing
end

dim = TwoDim(15.0f0, 700)

n = 5
μ = zeros(Float32, n, 2)
μ[1, :] .= [-10.0f0, 0.0f0]
μ[2, :] .= [-10.0f0, -1.0f0]
μ[3, :] .= [-10.0f0, 1.0f0]
μ[4, :] .= [-10.0f0, -2.0f0]
μ[5, :] .= [-10.0f0, 2.0f0]

σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.5f0

pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 100,
    actions = 20))

policy = RandomDesignPolicy(action_space(env))
render!(policy, env, Float32(env.actions) * 0.5f0)

# a = policy(env)
# tspan, interp, u_tot = env(a)
# s = state(env)
# wave = s.wave

# fig = Figure()
# ax1 = Axis(fig[1, 1], aspect = 1.0f0)
# ax2 = Axis(fig[1, 2], aspect = 1.0f0)
# ax3 = Axis(fig[1, 3], aspect = 1.0f0)

# heatmap!(ax1, wave[:, :, 1], colormap = :ice)
# heatmap!(ax2, wave[:, :, 2], colormap = :ice)
# heatmap!(ax3, wave[:, :, 3], colormap = :ice)
# save("wave.png", fig)