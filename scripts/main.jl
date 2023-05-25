using BSON
using Flux
using Flux: flatten, destructure, Scale
using ReinforcementLearning
using FileIO
using CairoMakie
using Metalhead
using Waves

include("improved_model.jl")

Flux.device!(0)

data_path = "data/main/episodes"

env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))

episode = EpisodeData(path = joinpath(data_path, "episode1/episode.bson"))  
states, actions, tspans, sigmas = prepare_data(episode, 10)

idx = 23
s = states[idx]
tspan = tspans[idx]
sigma = sigmas[idx]

nfreq = 6
h_size = 512
n_h = 2
activation = gelu

latent_dim = OneDim(15.0f0, 512)
wave_encoder = build_hypernet_wave_encoder(
    latent_dim = latent_dim,
    input_layer = TotalWaveInput(),
    nfreq = nfreq,
    h_size = h_size,
    activation = activation
) |> gpu

design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, n_h, activation, latent_dim)

@time c = design_encoder(s.design, policy(env)) |> cpu
@time uvf = wave_encoder(s |> gpu) |> cpu

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[2, 1])
ax4 = Axis(fig[2, 2])

lines!(ax1, latent_dim.x, uvf[:, 1])
lines!(ax2, latent_dim.x, uvf[:, 2])
lines!(ax3, latent_dim.x, uvf[:, 3])
lines!(ax4, latent_dim.x, c)

save("latent_$activation.png", fig)
