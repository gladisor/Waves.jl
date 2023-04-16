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
    σ = dyn.pml * dyn.pml_scale

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

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100
tspan = build_tspan(ti, dt, steps)
tf = tspan[end]

dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
grad = build_gradient(dim)
bc = dirichlet(dim)
pml = build_pml(dim, 2.0f0, 20000.0f0)
wave = build_wave(dim, fields = 6)

pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)
wave = pulse(wave)

initial = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])
design = DesignInterpolator(initial)

env = ScatteredWaveEnv(
    wave, wave,
    SplitWavePMLDynamics(design, dim, grid, ambient_speed, grad, bc, pml),
    SplitWavePMLDynamics(nothing, dim, grid, ambient_speed, grad, bc, pml),
    zeros(Float32, steps + 1),
    0,
    dt,
    steps) |> gpu

# states = ScatteredWaveEnvState[]
# actions = Scatterers[]
# sigma = []

# for i in 1:10
#     action = Scatterers([0.75f0 * randn(Float32) 0.75f0 * randn(Float32)], [0.0f0], [0.0f0])

#     push!(states, cpu(state(env)))
#     push!(actions, action)

#     @time env(gpu(action))

#     push!(sigma, cpu(env.σ))
# end

# data = Flux.DataLoader((states, actions, sigma), shuffle = true)

# latent_dim = OneDim(grid_size, 1024)
# latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 10000.0f0)

# function main()

#     model = WaveControlModel(
#         WaveEncoder(6, 32, 2, relu),
#         DesignEncoder(2 * length(vec(initial)), 128, 1024, relu),
#         Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
#         Chain(Flux.flatten, Dense(3072, 512, relu), Dense(512, 512, relu), Dense(512, 1), vec)) |> gpu

#     opt_state = Optimisers.setup(Optimisers.Adam(1e-5), model)

#     for epoch in 1:20
#         train_loss = 0.0f0
#         for sample in data
#             s, a, σ = gpu.(sample)

#             for i in axes(s, 1)
#                 loss, back = Flux.pullback(_model -> mse(σ[i], _model(s[i].wave_total, s[i].design, a[i])), model)
#                 gs = back(one(loss))[1]
#                 opt_state, model = Optimisers.update(opt_state, model, gs)
#                 train_loss += loss
#             end
#         end

#         println(train_loss / length(data))
#     end

#     return model
# end

# model = main()

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(model.iter.dynamics.pml))
# save("pml.png", fig)

# model(states[1].total, states[1].design, actions[1])

# sigma_pred = [cpu(model(gpu(states[i].wave_total), gpu(states[i].design), gpu(actions[i]))) for i in axes(states, 1)]

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(sigma_pred)
# save("sigma.png", fig)