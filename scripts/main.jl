using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves
include("improved_model.jl")
include("plot.jl")

Flux.device!(1)
data_path = "data/main/episodes"
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))
episode = EpisodeData(path = joinpath(data_path, "episode1/episode.bson"))  
states, actions, tspans, sigmas = prepare_data(episode, 1)

idx = 23
s = states[idx]
a = actions[idx]
tspan = tspans[idx]
sigma = sigmas[idx]

nfreq = 3
h_size = 256
activation = leakyrelu

latent_dim = OneDim(15.0f0, 512)

wave_encoder = build_hypernet_wave_encoder(
    latent_dim = latent_dim,
    input_layer = TotalWaveInput(),
    nfreq = nfreq,
    h_size = h_size,
    activation = activation
    )

design_encoder = HypernetDesignEncoder(
    env.design_space, 
    action_space(env), 
    nfreq, 
    h_size, 
    activation, 
    latent_dim
    )

dynamics = LatentDynamics(
    latent_dim,
    ambient_speed = env.total_dynamics.ambient_speed,
    freq = env.total_dynamics.source.freq,
    pml_width = 5.0f0,
    pml_scale = 10000.0f0
    )

iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

mlp = Chain( ## normalization in these layers seems to harm?
    flatten,
    Dense(4 * size(latent_dim, 1), h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, 1),
    vec
    )

model = ScatteredEnergyModel(
    wave_encoder, 
    design_encoder, 
    latent_dim, 
    iter, 
    env.design_space,
    mlp
    ) |> gpu

train_data = Flux.DataLoader(prepare_data(episode, 2), shuffle = true, batchsize = 10)

batch = gpu(first(train_data))
states, actions, tspans, sigmas = batch
overfit!(model, states, actions, tspans, sigmas, 5e-6)