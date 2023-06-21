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

Flux.CUDA.allowscalar(false)

"""
Assumes x1 and x2 have the same second dimention.

x1: (j, n)
x2: (k, n)
"""
function compute_isometry_loss(x1::AbstractMatrix{Float32}, x2::AbstractMatrix{Float32})
    loss = 0.0f0

    for i in axes(x1, 2)
        for j in axes(x1, 2)
            loss += sqrt(sum(abs2.(x1[:, i] .- x1[:, j]))) - sqrt(sum(abs2.(x2[:, i] .- x2[:, j])))
        end
    end

    return loss
end

Flux.device!(0)
# main_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset"
# main_path = "data/triple_ring_dataset"
main_path = "data/full_state_single_adjustable_radii"
data_path = joinpath(main_path, "episodes")

println("Loading env")
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Loading data")
@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:3])
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 3:4])

nfreq = 6
h_size = 256
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 1024
horizon = 1
wave_input_layer = TotalWaveInput()
batchsize = 5
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

model = ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp) |> gpu
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
# val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

states, actions, tspans, sigmas = gpu(first(train_loader))
s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]

model(s, a)

# path = mkpath("results/overfit/instance_norm_cnn_decoder_batchsize=$batchsize")
# model = overfit(model, states, actions, tspans, sigmas, lr, 100, path = path);
# # model = overfit(model, s, a, t, sigma, lr, 100, path = path);
