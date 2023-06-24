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

function latent_consistency_data(episode::EpisodeData, horizon::Int)
    states = Vector{WaveEnvState}[]
    actions = Vector{<:AbstractDesign}[]
    tspans = AbstractMatrix{Float32}[]
    sigmas = AbstractMatrix{Float32}[]

    n = horizon - 1

    for i in 1:(length(episode) - n - 1)
        boundary = i + n
        
        push!(states, episode.states[i:boundary + 1])
        push!(actions, episode.actions[i:boundary])
        push!(tspans, hcat(episode.tspans[i:boundary]...))
        push!(sigmas, hcat(episode.sigmas[i:boundary]...))
    end

    return (states, actions, tspans, sigmas)
end

function latent_consistency_data(data::Vector{EpisodeData}, horizon::Int)
    return vcat.(latent_consistency_data.(data, horizon)...)
end

function latent_consistency_and_reconstruction_loss(model::ScatteredEnergyModel, s::Vector{WaveEnvState}, a::DesignSequence, sigma::AbstractMatrix)

    uvf = model.wave_encoder(s)
    c = model.design_encoder(s[1], a)
    z = flatten_repeated_last_dim(generate_latent_solution(model, uvf[:, :, 1], c))

    z_pred = uvf[:, [1, 2], 2:end]
    z_true = z[:, [1, 2], 2:model.iter.steps:end]

    l_con_numerator = sqrt.(
        sum((z_pred .- z_true) .^ 2, ## sum of squares
        dims = (1, 2) ## sum along spacial and channel dimention, leave sequence un-summed
        )
    )

    ## adding a small epsilon to denominator to prevent zero division
    l_con_denominator = sqrt.(sum(z_true .^ 2, dims = (1, 2))) .+ 1f-5
    l_con = sum(l_con_numerator ./ l_con_denominator)

    sigma_pred = vec(model.mlp(Flux.unsqueeze(z, dims = ndims(z) + 1)))
    sigma_true = flatten_repeated_last_dim(sigma)

    l_rec = Flux.mse(sigma_pred, sigma_true)

    return l_rec + l_con 
end

function latent_consistency_and_reconstruction_loss(model::ScatteredEnergyModel, states::Vector{Vector{WaveEnvState}}, actions::Vector{<: DesignSequence}, sigmas::Vector{<: AbstractArray})
    return Flux.mean(latent_consistency_and_reconstruction_loss.([model], states, actions, sigmas))
end

function reconstruction_loss(model::ScatteredEnergyModel, s, a, sigma)
    y = flatten_repeated_last_dim(sigma)
    return Flux.mse(model(s, a,), y)
end

Flux.device!(0)

main_path = "/scratch/cmpe299-fa22/tristan/data/design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")

println("Loading Environment")
env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Load Data") ## full dataset (too big)
# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:1])
# @time val_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 2:3])

println("Declaring Hyperparameters")
nfreq = 50
h_size = 256 ## try increase (memory issues :( sad)
activation = leakyrelu
latent_grid_size = 15.0f0
latent_elements = 512
horizon = 2
wave_input_layer = TotalWaveInput()
batchsize = 5
pml_width = 10.0f0
pml_scale = 10000.0f0
lr = 1e-4
decay_rate = 1.0f0
k_size = 2
steps = 20
epochs = 200
loss_func = Flux.mse

MODEL_PATH = mkpath(joinpath(main_path, "models/gs=$(latent_grid_size)_ele=$(latent_elements)_horizon=$(horizon)_width=$(pml_width)_lr=$(lr)"))
println(MODEL_PATH)

println("Initializing Model Components")
latent_dim = OneDim(latent_grid_size, latent_elements)
wave_encoder = build_split_hypernet_wave_encoder(latent_dim = latent_dim, input_layer = wave_input_layer, nfreq = nfreq, h_size = h_size, activation = activation)
design_encoder = HypernetDesignEncoder(env.design_space, action_space(env), nfreq, h_size, activation, latent_dim)
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)
mlp = Chain(reshape_latent_solution, build_full_cnn_decoder(latent_elements, h_size, k_size, activation))
println("Constructing Model")
model = gpu(ScatteredEnergyModel(wave_encoder, design_encoder, latent_dim, iter, env.design_space, mlp))

println("Initializing DataLoaders")
train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
val_loader = DataLoader(prepare_data(val_data, horizon), shuffle = true, batchsize = batchsize)

opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(), Optimisers.Adam(lr))
states, actions, tspans, sigmas = gpu(first(train_loader))
s, a, t, sigma = states[1], actions[1], tspans[1], sigmas[1]
overfit(model, states, actions, tspans, sigmas, 100, opt = opt, path = "coefs")

# model.design_encoder(s, a)
# visualize!(model, s, a, t, sigma, path = "")

# y = cpu(model.wave_encoder(s))

# fig = Figure()
# ax1 = Axis(fig[1, 1])
# ax2 = Axis(fig[1, 2])
# ax3 = Axis(fig[2, 1])
# # ax4 = Axis(fig[2, 2])

# lines!(ax1, latent_dim.x, y[:, 1])
# lines!(ax2, latent_dim.x, y[:, 2])
# lines!(ax3, latent_dim.x, y[:, 3])
# # lines!(ax3, latent_dim.x, y[:, 4])
# save("n.png", fig)

# println("Training")
# train_loop(
#     model,
#     loss_func = loss_func,
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
#     path = MODEL_PATH,
#     opt = opt
#     )