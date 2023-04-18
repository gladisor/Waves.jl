using Flux
using Flux: Recur, pullback
using ChainRulesCore

using Interpolations
using Interpolations: Extrapolation
using CairoMakie
using Optimisers
using BSON
using ReinforcementLearning
using IntervalSets
using LinearAlgebra
using Waves

include("../src/dynamics.jl")
include("env.jl")
include("plot.jl")
include("wave_control_model.jl")

Flux.trainable(config::Scatterers) = (;config.pos,)

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100
tf = ti + steps * dt

dim = TwoDim(grid_size, elements)
pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)
initial_design = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])

env = ScatteredWaveEnv(
    dim,
    initial_condition = gpu(pulse),
    design = initial_design,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    reset_design = d -> random_pos(d, 3.0f0),
    action_space = Waves.design_space(initial_design, 1.0f0),
    dt = dt,
    integration_steps = steps,
    max_steps = 1000) |> gpu

model = BSON.load("model.bson")[:model] |> gpu
s = state(env)

wave = gpu(s.wave_total)
design = gpu(s.design)
control = build_control_sequence(initial_design, 1)
action = gpu(control[1])

opt_state = Optimisers.setup(Optimisers.Adam(1e-2), action)

z_wave = model.wave_encoder(wave)
f = a -> sum(model.mlp(model.iter(hcat(z_wave, z_wave * 0.0f0, model.design_encoder(design, a))))) + norm(vec(a))

for i in 1:10
    sigma, back = pullback(f, action)
    gs = back(one(sigma))[1]
    opt_state, action = Optimisers.update(opt_state, action, gs)
    println(sigma)
end

env(action)

s = state(env)
wave = gpu(s.wave_total)
design = gpu(s.design)
control = build_control_sequence(initial_design, 1)
action = gpu(control[1])

opt_state = Optimisers.setup(Optimisers.Adam(1.0f0), action)

z_wave = model.wave_encoder(wave)
f = a -> sum(model.mlp(model.iter(hcat(z_wave, z_wave * 0.0f0, model.design_encoder(design, a))))) + norm(vec(a)) * 0.001

for i in 1:20
    sigma, back = pullback(f, action)
    gs = back(one(sigma))[1]
    opt_state, action = Optimisers.update(opt_state, action, gs)
    println(sigma)
end

env(action)
println(sum(env.σ))
