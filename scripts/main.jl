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

function compute_gradient(model::ScatteredEnergyModel, states::Vector{WaveEnvState}, actions::Vector{Vector{AdjustableRadiiScatterers}}, sigma::Vector{<:AbstractArray}, loss_func::Function)
    loss, back = Flux.pullback(_model -> sum(loss_func.(_model.(states, actions), sigmas)), model)
    return loss, back(one(loss))[1]
end

function overfit!(model::ScatteredEnergyModel, states, actions, tspans, sigmas, lr)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, model)

    for i in 1:200
        loss, back = Flux.pullback(_model -> sum(Flux.mse.(_model.(states, actions), sigmas)), model)
        gs = back(one(loss))[1]
        println(loss)
        opt_state, model = Optimisers.update(opt_state, model, gs)
    end

    for (i, (s, a, tspan, sigma)) in enumerate(zip(states, actions, tspans, sigmas))
        @time visualize!(model, s, a, tspan, sigma, path = mkpath("$i"))
    end
end

Flux.device!(0)

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

nfreq = 10
h_size = 256
n_h = 4
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
    n_h, 
    activation, 
    latent_dim
    )

dynamics = LatentDynamics(
    latent_dim,
    ambient_speed = env.total_dynamics.ambient_speed,
    freq = env.total_dynamics.source.freq,
    pml_width = 5.0f0,
    pml_scale = 10000.0f0)

iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

mlp = Chain( ## normalization in these layers seems to harm?
    flatten,
    Dense(4 * size(latent_dim, 1), h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, 1),
    vec)

model = ScatteredEnergyModel(
    wave_encoder, 
    design_encoder, 
    latent_dim, 
    iter, 
    env.design_space,
    mlp) |> gpu

train_data = Flux.DataLoader(prepare_data(episode, 2), shuffle = true, batchsize = 10)

batch = gpu(first(train_data))
states, actions, tspans, sigmas = batch
overfit!(model, states, actions, tspans, sigmas, 5e-6)

# loss, gs = compute_gradient(model, s, a, sigma, Flux.mse)

# function recursive_add(t1::NamedTuple, t2::NamedTuple)
#     return (k => recursive_add(t1[k], t2[k]) for k in keys(t1))
# end

# added_gs = recursive_add(gs, gs)

# function recursive_print(d::NamedTuple)
#     for f in keys(d)
#         println(f)
#         recursive_print(d[f])
#     end
# end

# function recursive_print(d::Tuple)
#     for ele in d
#         recursive_print(ele)
#     end
# end

# function recursive_print(d::AbstractArray)
#     println("Found gradient")
#     return
# end

# function recursive_print(d::Float32)
#     println("Found float")
#     return
# end

# function recursive_print(::Nothing)
#     return nothing
# end

# recursive_print(gs)