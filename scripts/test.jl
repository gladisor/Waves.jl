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

struct Hypernet
    dense::Dense
    re::Optimisers.Restructure
    domain::FrequencyDomain
end

Flux.@functor Hypernet

function Hypernet(in_size::Int, base::Chain, domain::FrequencyDomain)
    ps, re = destructure(base)
    dense = Dense(in_size, length(ps), bias = false)
    return Hypernet(dense, re, domain)
end

function (hypernet::Hypernet)(x::AbstractMatrix{Float32})
   models = hypernet.re.(eachcol(hypernet.dense(x)))
   #return cat([hypernet.domain(m) for m in models]..., dims = 3)
   return [hypernet.domain(m) for m in models]
end

struct SophonWaveEncoder
    input_layer::WaveInputLayer
    layers::Chain
    scale::AbstractVector{Float32}
end

Flux.@functor SophonWaveEncoder
Flux.trainable(wave_encoder::SophonWaveEncoder) = (;wave_encoder.layers)

function SophonWaveEncoder(latent_dim::OneDim, input_layer::WaveInputLayer, nfreq::Int, h_size::Int, activation::Function; ambient_speed::Float32 = WATER)

    embedder = Chain(
        Dense(2 * nfreq, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, 3, tanh),
        )

    layers = Chain(
        MaxPool((4, 4)),
        ResidualBlock((3, 3), 1, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, 128, activation),
        GlobalMaxPool(),
        flatten,
        NormalizedDense(128, h_size, activation),
        Hypernet(h_size, embedder, FrequencyDomain(latent_dim, nfreq)),
        )

    scale = [1.0f0, 1.0f0 / ambient_speed, 1.0f0]
    return SophonWaveEncoder(input_layer, layers, scale)
end

function (model::SophonWaveEncoder)(s::WaveEnvState)
    x = model.layers(model.input_layer(s))
end

function (model::SophonWaveEncoder)(s::Vector{WaveEnvState})
    x = model.layers(cat(model.input_layer.(s)..., dims = 4))
end

struct SophonDesignEncoder
    design_space::DesignSpace
    action_space::DesignSpace
    layers::Chain
end

Flux.@functor SophonDesignEncoder

function SophonDesignEncoder(
    latent_dim::OneDim,
    design_space::DesignSpace,
    action_space::DesignSpace,
    nfreq::Int,
    h_size::Int,
    activation::Function)

    embedder = Chain(
        Dense(2 * nfreq, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation),
        Dense(h_size, 1, sigmoid))

    in_size = length(vec(design_space.low)) + length(vec(action_space.low))

    layers = Chain(
        NormalizedDense(in_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        NormalizedDense(h_size, h_size, activation),
        Hypernet(h_size, embedder, FrequencyDomain(latent_dim, nfreq)),
        )

    return SophonDesignEncoder(design_space, action_space, layers)
end

# function (design_encoder::SophonDesignEncoder)(d::AbstractDesign, a::AbstractDesign)    
#     d_norm = (vec(d) .- vec(design_encoder.design_space.low)) ./ (vec(design_encoder.design_space.high) .- vec(design_encoder.design_space.low) .+ EPSILON)
#     a_norm = (vec(a) .- vec(design_encoder.action_space.low)) ./ (vec(design_encoder.action_space.high) .- vec(design_encoder.action_space.low) .+ EPSILON)
#     return design_encoder.design_space(d, a), design_encoder.layers(vcat(d_norm, a_norm)[:, :])
# end

# function (design_encoder::SophonDesignEncoder)(d::AbstractDesign, a::Vector{ <: AbstractDesign})
#     recur = Recur(design_encoder, d)
#     return hcat([recur(action) for action in a]...)
# end

# function (design_encoder::SophonDesignEncoder)(states::Vector{WaveEnvState}, actions::Vector{<: Vector{<: AbstractDesign}})
#     designs = [s.design for s in states]
#     hcat(vec.(designs)...)
# end

function (design_encoder::SophonDesignEncoder)(d::AbstractDesign, a::AbstractDesign)
    next_design = design_encoder.design_space(d, a)
    return next_design, next_design
end

function (design_encoder::SophonDesignEncoder)(d::AbstractDesign, a::Vector{<: AbstractDesign})
    recur = Recur(design_encoder, d)
    designs = [recur(action) for action in a]

    x = vcat(
            hcat(vec.(designs)...), 
            hcat(vec.(a)...)
        )

    vcat(design_encoder.layers(x)...)'
end


Flux.device!(0)
main_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset"
data_path = joinpath(main_path, "episodes")

env = BSON.load(joinpath(main_path, "env.bson"))[:env] |> gpu
dim = cpu(env.dim)
reset!(env)

policy = RandomDesignPolicy(action_space(env))

println("Load Train Data")
@time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:2])

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
latent_dim = OneDim(latent_grid_size, latent_elements)

wave_encoder = gpu(SophonWaveEncoder(latent_dim, wave_input_layer, nfreq, h_size, activation))
design_encoder = gpu(SophonDesignEncoder(latent_dim, env.design_space, action_space(env), nfreq, h_size, activation))
dynamics = LatentDynamics(latent_dim, ambient_speed = env.total_dynamics.ambient_speed, freq = env.total_dynamics.source.freq, pml_width = pml_width, pml_scale = pml_scale)
iter = Integrator(runge_kutta, dynamics, 0.0f0, env.dt, env.integration_steps)

train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)

states, actions, tspans, sigmas = gpu(first(train_loader))

idx = 1
s, a, t, sigma = states[idx], actions[idx], tspans[idx], sigmas[idx]
# uvf = wave_encoder(states)
# c = design_encoder(s.design, a[1])
# c = design_encoder(s.design, a)
c, back = Flux.pullback(_a -> design_encoder(s.design, _a), a)
gs = back(c)[1]