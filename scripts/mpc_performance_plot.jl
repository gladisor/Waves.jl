using Waves
using BSON
using CairoMakie

include("improved_model.jl")
include("plot.jl")

Flux.CUDA.allowscalar(false)


main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")

## random
@time random_episodes = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 1:6])
data = prepare_data.(random_episodes, length(random_episodes[1]))
_, _, _, random_sigmas = zip(data...)
flatten_random_sigmas = hcat(map(x -> flatten_repeated_last_dim(x[1]), random_sigmas)...)
random_avg = vec(Flux.mean(flatten_random_sigmas, dims = 2))

## mpc
@time mpc_episodes = Vector{EpisodeData}([EpisodeData(path = "mpc_results/mpc_episode$i.bson") for i in 1:6])
data = prepare_data.(mpc_episodes, length(mpc_episodes[1]))
_, _, t, mpc_sigmas = zip(data...)

flattened_t = flatten_repeated_last_dim(t[1][1])
flatten_mpc_sigmas = hcat(map(x -> flatten_repeated_last_dim(x[1]), mpc_sigmas)...)
mpc_avg = vec(Flux.mean(flatten_mpc_sigmas, dims = 2))

## mpc horizon 30
@time mpc_episodes = Vector{EpisodeData}([EpisodeData(path = "mpc_results/mpc_episode_horizon=30_$i.bson") for i in 1:6])
data = prepare_data.(mpc_episodes, length(mpc_episodes[1]))
_, _, t, mpc_sigmas = zip(data...)

flattened_t = flatten_repeated_last_dim(t[1][1])
flatten_mpc_sigmas = hcat(map(x -> flatten_repeated_last_dim(x[1]), mpc_sigmas)...)
mpc_avg_horizon_30 = vec(Flux.mean(flatten_mpc_sigmas, dims = 2))

fig = Figure()
ax = Axis(
    fig[1, 1],
    title = "Reduction of Scattered Energy With Model Predictive Control",
    xlabel = "Time (s)",
    ylabel = "Scattered Energy",
)

lines!(ax, flattened_t, random_avg, label = "Random")
lines!(ax, flattened_t, mpc_avg, label = "MPC")
lines!(ax, flattened_t, mpc_avg_horizon_30, label = "horizon = 30")

axislegend(ax, position = :rb)
save("random.png", fig)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, t, random_avg, label = "Random")
# lines!(ax, t, mpc_avg, label = "MPC")
# save("random.png", fig)