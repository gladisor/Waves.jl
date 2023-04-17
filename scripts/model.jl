using CairoMakie
using Flux
using Flux: unbatch, mse
Flux.CUDA.allowscalar(false)
using Optimisers

using Flux: pullback
using ChainRulesCore

using Interpolations
using Interpolations: Extrapolation
using Waves

include("plot.jl")
using Flux: Params, Recur
using Waves: speed
include("../src/dynamics.jl")

using ReinforcementLearning
using IntervalSets
include("env.jl")

struct LatentPMLWaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    pml_scale::Float32

    pml::AbstractArray
    grad::AbstractMatrix
    bc::AbstractArray
end

Flux.@functor LatentPMLWaveDynamics
Flux.trainable(dyn::LatentPMLWaveDynamics) = (;dyn.pml)

function LatentPMLWaveDynamics(dim::AbstractDim; ambient_speed::Float32, pml_scale::Float32)
    pml = zeros(Float32, size(dim)...)
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    return LatentPMLWaveDynamics(ambient_speed, pml_scale, pml, grad, bc)
end

function (dyn::LatentPMLWaveDynamics)(u::AbstractMatrix{Float32}, t::Float32)
    U = u[:, 1]
    V = u[:, 2]
    C = u[:, 3] * dyn.ambient_speed

    ∇ = dyn.grad
    σ = dyn.pml * dyn.pml_scale .^ 2

    du = C .^ 2 .* ∂x(∇, V) .- σ .* U
    dv = ∂x(∇, U) .- σ .* V
    dc = C * 0.0f0

    return hcat(dyn.bc .* du, dv, dc)
end

struct WaveControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function encode(model::WaveControlModel, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    return hcat(model.wave_encoder(wave), model.design_encoder(design, action))
end

function (model::WaveControlModel)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    zi = encode(model, wave, design, action)
    z = model.iter(zi)
    return model.mlp(z)
end

function build_control_sequence(action::AbstractDesign, steps::Int)
    return [zero(action) for i in 1:steps]
end

function build_mpc_cost(model::WaveControlModel, s::ScatteredWaveEnvState, control_sequence::Vector{ <: AbstractDesign})
    cost = 0.0f0

    d1 = s.design
    c1 = model.design_encoder(d1, control_sequence[1])
    z1 = hcat(model.wave_encoder(s.wave_total), c1)
    z = model.iter(z1)
    cost = cost + sum(model.mlp(z))

    d2 = d1 + control_sequence[1]
    c2 = model.design_encoder(d2, control_sequence[2])
    z2 = hcat(z[:, 1:2, end], c2)
    z = model.iter(z2)
    cost = cost + sum(model.mlp(z))

    d3 = d2 + control_sequence[2]
    c3 = model.design_encoder(d3, control_sequence[3])
    z3 = hcat(z[:, 1:2, end], c3)
    z = model.iter(z3)
    cost = cost + sum(model.mlp(z))

    return cost
end

Flux.@functor Scatterers
Flux.trainable(config::Scatterers) = (;config.pos,)

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100
tspan = build_tspan(ti, dt, steps)
tf = tspan[end]

dim = TwoDim(grid_size, elements)
pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)
initial_design = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])

env = ScatteredWaveEnv(
    dim,
    initial_condition = pulse,
    design = initial_design,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    reset_design = d -> random_pos(d, 0.5f0),
    action_space = Waves.design_space(initial_design, 0.75f0),
    max_steps = 1000
    ) |> gpu

policy = RandomDesignPolicy(action_space(env))

states = ScatteredWaveEnvState[]
actions = AbstractDesign[]
sigmas = []

for episode in 1:1
    reset!(env)

    while !is_terminated(env)
        action = policy(env)
        @time env(action)

        push!(states, state(env))
        push!(actions, action)
        push!(sigmas, env.σ)
    end
end

latent_dim = OneDim(grid_size, 1024)
latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 10000.0f0)

model = WaveControlModel(
    WaveEncoder(6, 32, 2, relu),
    DesignEncoder(2 * length(vec(initial_design)), 128, 1024, relu),
    Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
    Chain(Flux.flatten, Dense(3072, 512, relu), Dense(512, 512, relu), Dense(512, 1), vec)) |> gpu

data = Flux.DataLoader((states, actions, sigmas), shuffle = true)

opt_state = Optimisers.setup(Optimisers.Adam(1e-5), model)

for (s, a, sigma) in data

    loss, back = pullback(_model -> mse(_model(gpu(s[1].wave_total), gpu(s[1].design), gpu(a[1])), gpu(sigma[1])), model)
    gs = back(one(loss))[1]
    opt_state, model = Optimisers.update(opt_state, model, gs)

    println(loss)
end


# s = gpu(state(env))
# control_sequence = gpu(build_control_sequence(initial_design, 2))

# opt_state = Optimisers.setup(Optimisers.Momentum(1e-5), control_sequence)

# for i in 1:10
#     cost, back = pullback(a -> build_mpc_cost(model, s, a), control_sequence)
#     gs = back(one(cost))[1]
#     opt_state, control_sequence = Optimisers.update(opt_state, control_sequence, gs)
#     println(cost)
# end