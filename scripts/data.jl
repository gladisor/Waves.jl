using Waves
using Flux
using ReinforcementLearning
using CairoMakie

dim = TwoDim(15.0f0, 700)

## beam
# n = 10
# μ = zeros(Float32, n, 2)
# μ[:, 1] .= -10.0f0
# μ[:, 2] .= range(-2.0f0, 2.0f0, n)
# σ = ones(Float32, n) * 0.3f0
# a = ones(Float32, n) * 0.3f0


## single pulse
μ = zeros(Float32, 1, 2)
μ[1, :] .= [-10.0f0, 0.0f0]
σ = [0.3f0]
a = [1.0f0]

pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 100,
    actions = 200))

policy = RandomDesignPolicy(action_space(env))
ep = generate_episode!(policy, env)
save(ep, "episode.bson")