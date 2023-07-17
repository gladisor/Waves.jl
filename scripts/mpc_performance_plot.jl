using Waves
using BSON
using CairoMakie

include("improved_model.jl")
include("plot.jl")

Flux.CUDA.allowscalar(false)

main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")

function load_and_average_sigma(paths::Vector{String})
    episodes = Vector{EpisodeData}([EpisodeData(path = path) for path in paths])
    data = prepare_data.(episodes, length(episodes[1]))
    _, _, t, sigmas = zip(data...)

    flattened_t = flatten_repeated_last_dim(t[1][1])
    flatten_sigmas = hcat(map(x -> flatten_repeated_last_dim(x[1]), sigmas)...)
    return flattened_t, vec(Flux.mean(flatten_sigmas, dims = 2))
end

## random
@time _, random_avg = load_and_average_sigma([joinpath(data_path, "episode$i/episode.bson") for i in 1:6])
@time t, pml_horizon_20_sigma_avg = load_and_average_sigma(["mpc_results/mpc_episode$i.bson" for i in 1:6])
@time t, no_pml_horizon_20_sigma_avg = load_and_average_sigma(["mpc_results/no_pml_mpc_episode_horizon=20_$i.bson" for i in 1:4])
@time t, pml_horizon_50_sigma_avg = load_and_average_sigma(["mpc_results/pml_mpc_episode_horizon=50_$i.bson" for i in 1:2])

fig = Figure()
ax = Axis(
    fig[1, 1],
    title = "Reduction of Scattered Energy With Model Predictive Control",
    xlabel = "Time (s)",
    ylabel = "Scattered Energy")

lines!(ax, t, random_avg, label = "Random")
lines!(ax, t, pml_horizon_20_sigma_avg, label = "Horizon = 20")
lines!(ax, t, pml_horizon_50_sigma_avg, label = "Horizon = 50")
axislegend(ax, position = :rb)
save("random.png", fig)