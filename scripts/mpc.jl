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

control = build_control_sequence(initial_design, 3)
action = control[1]

sigma_pred = model(gpu(s.wave_total), gpu(s.design), gpu(action))