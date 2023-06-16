using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves
using Statistics: mean, std
include("improved_model.jl")
include("plot.jl")

Flux.device!(1)
# main_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset"
main_path = "data/triple_ring_dataset"
data_path = joinpath(main_path, "episodes")

println("Loading env")
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

# println("Loading data")
# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 3:4])

nfreq = 6
h_size = 256
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 1024
horizon = 1
wave_input_layer = TotalWaveInput()
batchsize = 10
pml_width = 10.0f0
pml_scale = 10000.0f0
lr = 5e-6
k_size = 2

latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

# using displacement and wavespeed seems to be pretty good
mlp = Chain(
    reshape_latent_solution,
    # build_mlp_decoder(latent_elements, h_size, activation)
    build_full_cnn_decoder(latent_elements, h_size, k_size, activation)
    )

# mlp = Chain(
#     z -> permutedims(z, (1, 3, 2, 4)),
#     z -> z[:, :, 1:4, :],

#     Conv((2, 1), 4 => h_size, activation, pad = SamePad()),
#     x -> pad_reflect(x, (0, 0, k_size - 1, 0)),
#     Conv((1, 2), h_size => h_size, activation),
#     x -> pad_reflect(x, (k_size - 1, 0, k_size - 1, 0)),
#     Conv((k_size, k_size), h_size => h_size, activation),
#     MaxPool((2, 1)),

#     Conv((2, 1), h_size => h_size, activation, pad = SamePad()),
#     x -> pad_reflect(x, (0, 0, k_size - 1, 0)),
#     Conv((1, 2), h_size => h_size, activation),
#     x -> pad_reflect(x, (k_size - 1, 0, k_size - 1, 0)),
#     Conv((k_size, k_size), h_size => h_size, activation),
#     MaxPool((2, 1)),


#     Conv((2, 1), h_size => h_size, activation, pad = SamePad()),
#     x -> pad_reflect(x, (0, 0, k_size - 1, 0)),
#     Conv((1, 2), h_size => h_size, activation),
#     x -> pad_reflect(x, (k_size - 1, 0, k_size - 1, 0)),
#     Conv((k_size, k_size), h_size => 1, activation),
#     MaxPool((2, 1)),
#     Dense(128, 1),
#     flatten
# )

model = ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp) |> gpu
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
# val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

# states, actions, tspans, sigmas = gpu(first(train_loader))
s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]

# model = gpu(BSON.load("/home/012761749/Waves.jl/data/triple_ring_dataset/models/full_cnn/pml_width=10.0_pml_scale=10000.0_k_size=2_latent_elements=1024/epoch_120/model.bson")[:model])
model = overfit(model, s, a, t, sigma, lr, 300)

# uvf = model.wave_encoder(s)[:, :, 1]
# y = flatten_repeated_last_dim(sigma)
# loss, back = Flux.pullback((_uvf, _a) -> Flux.mse(model.mlp(Flux.unsqueeze(flatten_repeated_last_dim(generate_latent_solution(model, _uvf, s.design, _a)), dims = 4)), y), uvf, a)
# gs = back(one(loss))[1]

# fig = Figure()
# ax1 = Axis(fig[1, 1])
# lines!(ax1, cpu(gs[:, 1]))

# ax2 = Axis(fig[1, 2])
# lines!(ax2, cpu(gs[:, 2]))

# ax3 = Axis(fig[2, 1])
# lines!(ax3, cpu(gs[:, 3]))
# save("gs.png", fig)

# z = flatten_repeated_last_dim(generate_latent_solution(model, uvf, s.design, a))
# render!(latent_dim, cpu(z), path = "vid.mp4")

# loss, back = Flux.pullback(m -> Flux.mse(m(s, a), y), model)
# gs = back(one(loss))[1]
# y = model(s, a)
# size(y)