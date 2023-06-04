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

Flux.device!(1)
main_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset"
data_path = joinpath(main_path, "episodes")

env = BSON.load(joinpath(main_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))

println("Load Train Data")
@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:150
    ])

println("Load Val Data")
@time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 151:200
    ])

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
MODEL_PATH = mkpath(joinpath(main_path, "models/Rerun/nfreq=$(nfreq)_hsize=$(h_size)_act=$(activation)_gs=$(latent_grid_size)_ele=$(latent_elements)_hor=$(horizon)_in=$(wave_input_layer)_bs=$(batchsize)_pml=$(pml_scale)_lr=$(lr)_decay=$(decay_rate)"))
println(MODEL_PATH)
steps = 20
epochs = 1000

latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

mlp = Chain( ## normalization in these layers seems to harm?
    flatten,
    ## transient wavespeed
    # Dense(5 * size(latent_dim, 1), h_size, activation),
    ## static wavespeed
    Dense(4 * size(latent_dim, 1), h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, 1),
    vec)

# struct BLSTM
#     forward
#     backward
#     dense::Dense
# end

# Flux.@functor BLSTM

# function BLSTM(in_size, out_size, activation)
#     forward = LSTM(in_size, out_size)
#     backward = LSTM(in_size, out_size)
#     dense = Dense(2 * out_size, out_size, activation)
#     return BLSTM(forward, backward, dense)
# end

# function (blstm::BLSTM)(x)
#     f = blstm.forward(x)
#     b = reverse(blstm.backward(reverse(x, dims = 3)), dims = 3)
#     return blstm.dense(vcat(f, b))
# end

# mlp = Chain(
#     flatten,
#     Dense(4 * size(latent_dim, 1), h_size, activation),
#     # Dense(h_size, h_size, activation),
#     # Dense(h_size, h_size, activation),
#     x -> x[:, :, :],
#     x -> permutedims(x, (1, 3, 2)),
#     BLSTM(h_size, h_size, activation),
#     BLSTM(h_size, h_size, activation),
#     BLSTM(h_size, 1, identity),
#     # Dense(h_size, h_size, activation),
#     # Dense(h_size, 1),
#     vec
# ) |> gpu

# mlp = Chain(
#     flatten,
#     Dense(4 * latent_elements, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, 1),
#     vec)

# mlp = Chain(
#     # x -> x[:, [1, 2], :],
#     flatten,
#     Dense(4 * latent_elements, h_size, activation),
#     Dense(h_size, h_size, activation),
#     x -> x[:, :, :],
#     x -> permutedims(x, (2, 1, 3)),
#     x -> pad_reflect(x, (2, 0)),
#     Conv((3,), h_size => h_size, activation),
#     x -> pad_reflect(x, (2, 0)),
#     Conv((3,), h_size => h_size, activation),
#     x -> pad_reflect(x, (2, 0)),
#     Conv((3,), h_size => h_size, activation),
#     Conv((1,), h_size => 1),
#     vec
# )

model = ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp) |> gpu
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

## investigate further if i can only include u and v fields in mlp
# states, actions, tspans, sigmas = gpu(first(train_loader))
# s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]
# zi = hcat(model.wave_encoder(s), model.design_encoder(s.design, a[1]))
# z = model.iter(zi)
# y = model.mlp(z)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(y))
# save("sigma.png", fig)
# overfit!(model, states, actions, tspans, sigmas, lr, 100)

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