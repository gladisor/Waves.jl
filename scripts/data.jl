using Waves
using Flux
using ReinforcementLearning
using CairoMakie
using BSON

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/"

dim = TwoDim(15.0f0, 700)

## beam
# n = 10
# μ = zeros(Float32, n, 2)
# μ[:, 1] .= -10.0f0
# μ[:, 2] .= range(-2.0f0, 2.0f0, n)
# σ = ones(Float32, n) * 0.3f0
# a = ones(Float32, n) * 0.3f0

## single pulse
μ = zeros(Float32, 2, 2)
μ[1, :] .= [-10.0f0, 0.0f0]
μ[2, :] .= [10.0f0, 0.0f0]
σ = [0.3f0, 0.3f0]
a = [1.0f0, 1.0f0]
pulse = build_normal(build_grid(dim), μ, σ, a)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = Source(pulse, 1000.0f0),
    integration_steps = 100,
    actions = 10))

policy = RandomDesignPolicy(action_space(env))
@time render!(policy, env, path = "vid.mp4")

# name =  "$(typeof(env.iter.dynamics))_" *
#         "$(typeof(env.design))_" *
#         "Pulse_" * 
#         "dt=$(env.dt)_" *
#         "steps=$(env.integration_steps)_" *
#         "actions=$(env.actions)_" *
#         "actionspeed=$(env.action_speed)_" *
#         "resolution=$(env.resolution)"

# path = mkpath(joinpath(DATA_PATH, name))
# BSON.bson(joinpath(path, "env.bson"), env = cpu(env))

# for i in 11:500
#     ep = generate_episode!(policy, env)
#     save(ep, joinpath(path, "episode$i.bson"))
# end