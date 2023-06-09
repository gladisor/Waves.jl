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

# println("Load Train Data")
# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:150
#     ])

# println("Load Val Data")
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 151:200
#     ])

# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])
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
MODEL_PATH = mkpath(joinpath(main_path, "models/CNN_DISPLACEMENT/nfreq=$(nfreq)_hsize=$(h_size)_act=$(activation)_gs=$(latent_grid_size)_ele=$(latent_elements)_hor=$(horizon)_in=$(wave_input_layer)_bs=$(batchsize)_pml=$(pml_scale)_lr=$(lr)_decay=$(decay_rate)"))
println(MODEL_PATH)
steps = 20
epochs = 1000

latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

# mlp = Chain( ## normalization in these layers seems to harm?
#     x -> reshape(x, (latent_elements * size(x, 2), :, size(x, ndims(x)))),
#     Dense(5 * size(latent_dim, 1), h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, 1),
#     flatten
#     )

k_size = 2
mlp = Chain(
    x -> x[:, [1, 2, 3, 4, 5], :, :],
    x -> reshape(x, (latent_elements * size(x, 2), :, size(x, ndims(x)))),

    Dense(5 * latent_elements, h_size, activation),
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

function overfit(model::ScatteredEnergyModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, t::AbstractMatrix{Float32}, sigma::AbstractMatrix{Float32})
    y = flatten_repeated_last_dim(sigma)
    opt = Optimisers.Adam(5e-6)
    opt_state = Optimisers.setup(opt, model)

    trainmode!(model)

    for i in 1:100

        Flux.reset!(model)
        loss, back = Flux.pullback(m -> Flux.mse(m(s, a), y), model)
        println("Update: $i, Loss: $loss")
        gs = back(one(loss))[1]
        opt_state, model = Optimisers.update(opt_state, model, gs)
    end

    Flux.reset!(model)
    testmode!(model)
    visualize!(model, s, a, t, sigma, path = "")
end

# function overfit(model::ScatteredEnergyModel, s::Vector{WaveEnvState}, a::Vector{<: Vector{<: AbstractDesign}}, t::Vector{ <: AbstractMatrix{Float32}}, sigma::Vector{ <: AbstractMatrix{Float32}})
#     y = flatten_repeated_last_dim(sigma)
#     opt = Optimisers.Adam(5e-6)
#     opt_state = Optimisers.setup(opt, model)

#     trainmode!(model)

#     for i in 1:100

#         Flux.reset!(model)
#         loss, back = Flux.pullback(m -> Flux.mse(m(s, a), y), model)
#         println("Update: $i, Loss: $loss")
#         gs = back(one(loss))[1]
#         opt_state, model = Optimisers.update(opt_state, model, gs)
#     end

#     Flux.reset!(model)
#     testmode!(model)
#     visualize!(model, s, a, t, sigma, path = "")
# end

model = ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp) |> gpu
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

# states, actions, tspans, sigmas = gpu(first(train_loader))
s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]
# overfit(model, s, a, t, sigma)
# ys = model(states, actions)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(ys[:, 1]))
# save("sigma.png", fig)

uvf = model.wave_encoder(s)
z = flatten_repeated_last_dim(generate_latent_solution(model, uvf[:, :, 1], s.design, a))
# render!(latent_dim, cpu(z), path = "vid.mp4")

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(
    ax, 
    latent_dim.x, 
    cpu(flatten_repeated_last_dim(t)),
    cpu(z[:, end - 1, :]),
    colormap = :ice,
    )

save("z.png", fig)


# y, back = Flux.pullback(m -> m(s, a), model)
# gs = back(y)[1]

# train_loop(
#     model,
#     loss_func = Flux.mse,
#     train_steps = steps,
#     train_loader = train_loader,
#     val_steps = steps,
#     val_loader = val_loader,
#     epochs = epochs,
#     lr = lr,
#     decay_rate = decay_rate,
#     latent_dim = latent_dim,
#     evaluation_samples = 1,
#     checkpoint_every = 10,
#     path = MODEL_PATH
#     )