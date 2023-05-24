using BSON
using Flux
using Flux: flatten, destructure
using ReinforcementLearning
using FileIO
using CairoMakie
using Metalhead
using Waves

include("improved_model.jl")


function build_hypernet_wave_encoder(;
        latent_dim::OneDim,
        nfreq::Int = 3,
        h_size::Int = 512,
        activation::Function = relu,
        input_layer::WaveInputLayer = TotalWaveInput(),
        )

    embedder = build_mlp(2 * nfreq, h_size, 2, 3, activation)
    ps, re = destructure(embedder)

    model = Chain(
        input_layer,
        MaxPool((4, 4)),
        ResidualBlock((3, 3), 1, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, 128, activation),
        GlobalMaxPool(),
        flatten,
        NormalizedDense(128, 512, activation),
        Dense(512, length(ps), bias = false),
        vec,
        re,
        FrequencyDomain(latent_dim, nfreq),
    )

    return model
end

struct HypernetDesignEncoder
    design_space::DesignSpace
    action_space::DesignSpace
    layers::Chain
end

Flux.@functor HypernetDesignEncoder

function HypernetDesignEncoder(
        design_space::DesignSpace, 
        action_space::DesignSpace, 
        h_size::Int, 
        n_h::Int, 
        activation::Function,
        latent_dim::OneDim)

    in_size = length(vec(design_space.low)) + length(vec(action_space.low))
    layers = build_mlp(in_size, h_size, n_h, h_size, activation)

    return HypernetDesignEncoder(design_space, action_space, layers)
end

function (model::HypernetDesignEncoder)(d::AbstractDesign, a::AbstractDesign)
    (d - model.design_space.low)
    x = vcat(vec(d), vec(a))
end

# function build_hypernet_design_encoder(;
#         latent_dim::OneDim,
#         nfreq::Int,
#         in_size::Int
#         h_size::Int,
#         n_h::Int,
#         activation::Function)

#     ## parameterizes a single function (wave speed)
#     embedder = build_mlp(2 * nfreq, h_size, 2, 1, activation)
#     ps, re = destructure(embedder)

#     encoder = Chain(
#         build_mlp(in_size, h_size, n_h, h_size, act),
#         LayerNorm(h_size),
#         act,
#         Dense(h_size, length(ps), bias = false),
#         re,
#         FrequencyDomain(dim, nfreq),
#         vec,
#         speed_activation,
#         )
# end

Flux.device!(0)

data_path = "data/main/episodes"

env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))

# include("plot.jl")
# @time render!(policy, env, path = "vid.mp4", seconds = env.actions * 0.5f0)

episode = EpisodeData(path = joinpath(data_path, "episode1/episode.bson"))  
states, actions, tspans, sigmas = prepare_data(episode, 10)

idx = 23
s = states[idx]
tspan = tspans[idx]
sigma = sigmas[idx]

nfreq = 2
h_size = 512
n_h = 2
activation = relu

latent_dim = OneDim(15.0f0, 512)
# model = build_hypernet_wave_encoder(
#     input_layer = TotalWaveInput(),
#     latent_dim = latent_dim,
#     nfreq = nfreq
# ) |> gpu

model = HypernetDesignEncoder(env.design_space, action_space(env), h_size, n_h, activation, latent_dim)
model(s.design, policy(env))

# @time y = model(s |> gpu) |> cpu

# fig = Figure()
# ax1 = Axis(fig[1, 1])
# ax2 = Axis(fig[1, 2])
# ax3 = Axis(fig[1, 3])

# lines!(ax1, latent_dim.x, y[:, 1])
# lines!(ax2, latent_dim.x, y[:, 2])
# lines!(ax3, latent_dim.x, y[:, 3])
# save("latent_total.png", fig)
