using Waves
using Flux
Flux.CUDA.allowscalar(false)
using Optimisers
using CairoMakie
using ReinforcementLearning
using BSON
using FileIO

dim = TwoDim(15.0f0, 700)
n = 10
μ = zeros(Float32, n, 2)
μ[:, 1] .= -10.0f0
μ[:, 2] .= range(-2.0f0, 2.0f0, n)

σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.3f0
pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 100,
    actions = 10))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env, path = "vid.mp4")
ep = generate_episode!(policy, env)

# horizon = 5
# data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = 2, shuffle = true, partial = false)
# s, a, t, y = gpu(Flux.batch.(first(data)))
# latent_dim = OneDim(15.0f0, 700)

# nfreq = 50
# in_size = 18
# h_size = 256
# activation = leakyrelu

# dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
# iter = gpu(Integrator(runge_kutta, dyn, env.dt))
# wave_encoder = gpu(build_wave_encoder(;latent_dim, nfreq, h_size))
# wave = wave_encoder(s)

# mlp = Chain(
#     Dense(in_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, nfreq, tanh),
#     Waves.SinWaveEmbedder(latent_dim, nfreq),
#     sigmoid
#     )

# de = gpu(DesignEncoder(env.design_space, mlp, env.integration_steps))

# C = de(s, a, t)
# F = gpu(Waves.SinusoidalSource(latent_dim, nfreq, 1000.0f0))
# θ = [C, F]