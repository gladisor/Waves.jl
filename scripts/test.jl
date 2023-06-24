using Waves
using CairoMakie
using Flux
using Flux: DataLoader, flatten, Recur
using Optimisers
using ReinforcementLearning
using BSON
include("improved_model.jl")
include("plot.jl")

main_path = "/scratch/cmpe299-fa22/tristan/data/design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")

# println("Loading Environment")
# env = BSON.load(joinpath(data_path, "env.bson"))[:env] |> gpu
# dim = cpu(env.dim)
# reset!(env)
# policy = RandomDesignPolicy(action_space(env))

gs = 15.0f0
elements = 1024
nfreq = 50
horizon = 1
batchsize = 5

# @time train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:1])
# train_loader = DataLoader(prepare_data(train_data, horizon), shuffle = true, batchsize = batchsize)
# states, actions, tspans, sigmas = gpu(first(train_loader))
# s, a, t, sigma = states[end], actions[end], tspans[end], sigmas[end]

dim = OneDim(gs, elements)
embedder = gpu(SinWaveEmbedder(dim, nfreq))
x = gpu(randn(Float32, nfreq, 1))
y = cpu(vec(embedder(x)))

# wave_encoder = gpu(build_split_hypernet_wave_encoder(latent_dim = dim, input_layer = TotalWaveInput(), nfreq = nfreq, h_size = 256, activation = leakyrelu))
# design_encoder = gpu(HypernetDesignEncoder(env.design_space, action_space(env), nfreq, 256, leakyrelu, dim))

# b = wave_encoder(states)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, dim.x, y)
save("ui.png", fig)