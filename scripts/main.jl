println("Importing Packages")
using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves

using Wandb, Logging

include("improved_model.jl")
include("plot.jl")

Flux.device!(0)

main_path = "/scratch/cmpe299-fa22/tristan/data/full_state_triple_ring_dataset"
# main_path = "data/full_state_single_adjustable_radii"
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Load Data")
## full dataset (too big)
# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:150])
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 151:200])

## for initial pretrain
# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:50])
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 51:60])

## for transfer learning
@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 25:75])
@time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 185:195])

println("Declaring Hyperparameters")
nfreq = 6
h_size = 256 ## try increase (memory issues :( sad)
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 1024
horizon = 5
wave_input_layer = TotalWaveInput()
batchsize = 5
pml_width = 10.0f0
pml_scale = 10000.0f0
lr = 5e-6
decay_rate = 1.0f0
k_size = 2
steps = 20
epochs = 200#1000

# MODEL_PATH = mkpath(joinpath(main_path, "models/no_norm_horizon=$horizon/batchsize=$(batchsize)_pml_off"))
MODEL_PATH = mkpath(joinpath(main_path, "models/transfer_learning/model_name_placeholder_horizon=$(horizon)"))
println(MODEL_PATH)

# model = gpu(BSON.load(joinpath(main_path, "models/no_norm_horizon=1/batchsize=5/epoch_1000/model.bson"))[:model])
# model = gpu(BSON.load("data/full_state_single_adjustable_radii/models/transfer_learning/model_name_placeholder/epoch_1000/model.bson")[:model])
model = gpu(BSON.load("/scratch/cmpe299-fa22/tristan/data/full_state_triple_ring_dataset/models/transfer_learning/model_name_placeholder_horizon=2/epoch_190/model.bson")[:model])

println("Initializing Model Components")
latent_dim = OneDim(latent_grid_size, latent_elements)
# wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
# design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
# dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
# iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

# ## using displacement and wavespeed seems to be pretty good
# mlp = Chain(
#     reshape_latent_solution,
#     build_full_cnn_decoder(latent_elements, h_size, k_size, activation)
#     )

println("Constructing Model")
# model = ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp) |> gpu
# model = ScatteredEnergyModel(model.wave_encoder, design_encoder, latent_dim, model.iter, env.design_space, model.mlp) |> gpu

println("Initializing DataLoaders")
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

states, actions, tspans, sigmas = gpu(first(train_loader))
s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]
# model = overfit(model, states, actions, tspans, sigmas, lr, 100, path = "");

println("Training")
train_loop(
    model,
    loss_func = Flux.mse,
    train_steps = steps,
    train_loader = train_loader,
    val_steps = steps,
    val_loader = val_loader,
    epochs = epochs,
    lr = lr,
    decay_rate = decay_rate,
    latent_dim = latent_dim,
    evaluation_samples = 1,
    checkpoint_every = 10,
    path = MODEL_PATH
    )