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

Flux.device!(0)

main_path = "data/single_cylinder_dataset"
data_path = joinpath(main_path, "episodes")

env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))

println("Load Train Data")
@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:1#40
    ])
println("Load Val Data")
@time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 2:3#41:52
    ])

nfreq = 6
h_size = 256
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 512
horizon = 3
wave_input_layer = TotalWaveInput()
batchsize = 10
pml_scale = 10000.0f0
lr = 1e-6
decay_rate = 1.0f0
MODEL_PATH = mkpath(joinpath(main_path, "models/nfreq=$(nfreq)_hsize=$(h_size)_act=$(activation)_gs=$(latent_grid_size)_ele=$(latent_elements)_hor=$(horizon)_in=$(wave_input_layer)_bs=$(batchsize)_pml=$(pml_scale)_lr=$(lr)_decay=$(decay_rate)"))

steps = 20
epochs = 1000

latent_dim = OneDim(latent_grid_size, latent_elements)

wave_encoder = build_hypernet_wave_encoder(
    latent_dim = latent_dim,
    input_layer = wave_input_layer,
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
    pml_scale = pml_scale
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
    mlp)# |> gpu

train_data = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
val_data = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

states, actions, tspans, sigmas = first(val_data) |> gpu

struct LatentSeparation
    base::ScatteredEnergyModel
    incident_encoder::Chain
end

Flux.@functor LatentSeparation

function (ls::LatentSeparation)(s::WaveEnvState, a::Vector{<: AbstractDesign})
    return ls.base(s, a)
end

incident_encoder = build_hypernet_wave_encoder(
    latent_dim = latent_dim,
    input_layer = IncidentWaveInput(),
    nfreq = nfreq,
    h_size = h_size,
    activation = activation
    )

ls = LatentSeparation(model, incident_encoder) |> gpu

ls.(states, actions)

# train_loop(
#     model,
#     loss_func = Flux.mse,
#     train_steps = steps,
#     train_loader = train_data,
#     val_steps = steps,
#     val_loader = val_data,
#     epochs = epochs,
#     lr = lr,
#     decay_rate = decay_rate,
#     latent_dim = latent_dim,
#     evaluation_samples = 1,
#     checkpoint_every = 1,
#     path = MODEL_PATH
#     )