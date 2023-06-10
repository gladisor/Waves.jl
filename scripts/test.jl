using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves
include("improved_model.jl")
include("plot.jl")

Flux.device!(2)
main_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset"
data_path = joinpath(main_path, "episodes")

env = BSON.load(joinpath(main_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))

@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 3:4])

nfreq = 6
h_size = 256
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 512
horizon = 5
wave_input_layer = TotalWaveInput()
batchsize = 10

pml_width = 5.0f0
pml_scale = 10000.0f0
lr = 5e-6
decay_rate = 1.0f0
steps = 20
epochs = 1000

latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

k_size = 2
mlp = Chain(
    x -> x[:, [1], :, :],
    x -> reshape(x, (latent_elements * size(x, 2), :, size(x, ndims(x)))),

    Dense(1 * latent_elements, h_size, activation),
    x -> permutedims(x, (2, 1, 3)),

    x -> pad_reflect(x, (k_size - 1, 0)),
    Conv((k_size,), h_size => h_size, activation),

    x -> pad_reflect(x, (k_size - 1, 0)),
    Conv((k_size,), h_size => h_size, activation),

    x -> pad_reflect(x, (k_size - 1, 0)),
    Conv((k_size,), h_size => h_size, activation),

    x -> pad_reflect(x, (k_size - 1, 0)),
    Conv((k_size,), h_size => h_size, activation),

    x -> pad_reflect(x, (k_size - 1, 0)),
    Conv((k_size,), h_size => h_size, activation),

    x -> pad_reflect(x, (k_size - 1, 0)),
    Conv((k_size,), h_size => 1),
    flatten
)

model = ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp) |> gpu
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
# val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

# states, actions, tspans, sigmas = gpu(first(train_loader))
s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]
overfit(model, states, actions, tspans, sigmas)

# uvf = model.wave_encoder(s)[:, :, 1]
# c = model.design_encoder(s.design, a[1])
# zi = hcat(uvf, c)
# z = model.iter(zi)

# c2 = model.design_encoder.layers(
#     normalize(
#         model.design_encoder.design_space(s.design, a[1]), 
#         model.design_encoder.design_space)
#     )

# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, cpu(z[:, end-1, :]), colormap = :ice)
# save("z.png", fig)