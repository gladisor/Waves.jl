println("Importing Packages")
using BSON
using Flux
using Flux: flatten, destructure, Scale, Recur, DataLoader
using ReinforcementLearning
using FileIO
using CairoMakie
using Optimisers
using Waves

Flux.CUDA.allowscalar(false)

include("improved_model.jl")
include("plot.jl")

Flux.device!(0)

main_path = "/scratch/cmpe299-fa22/tristan/data/design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

"""
General comments:

- Using information about wave and also about wavespeed function in the decoder is
critical to ensure smooth solution.

- Need a high enough number of frequencies

"""

println("Declaring Hyperparameters")
nfreq = 200
h_size = 256
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 512
horizon = 20
wave_input_layer = TotalWaveInput()
batchsize = 64

pml_width = 10.0f0
pml_scale = 0.0f0 #10000.0f0
lr = 1e-4
decay_rate = 1.0f0
k_size = 2
steps = 20
epochs = 500
loss_func = Flux.mse

MODEL_PATH = mkpath(joinpath(main_path, "models/SinWaveEmbedderV9/dropout_horizon=$(horizon)_nfreq=$(nfreq)_pml=$(pml_scale)_lr=$(lr)_batchsize=$(batchsize)"))
println(MODEL_PATH)

println("Initializing Model Components")
latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)
mlp = build_scattered_wave_decoder(latent_elements, h_size, k_size, activation)
println("Constructing Model")
model = gpu(ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp))

println("Initializing DataLoaders")
@time begin
    train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:180])
    val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 181:200])
    # train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])
    # val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 3:4])
    train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize, partial = false)
    val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize, partial = false)
end

opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(), Optimisers.Adam(lr))

println("Training")
train_loop(
    model,
    loss_func = loss_func,
    train_steps = steps,
    train_loader = train_loader,
    val_steps = steps,
    val_loader = val_loader,
    epochs = epochs,
    lr = lr,
    decay_rate = decay_rate,
    latent_dim = latent_dim,
    evaluation_samples = 15,
    checkpoint_every = 10,
    path = MODEL_PATH,
    opt = opt
    )