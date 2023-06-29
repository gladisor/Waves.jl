using Waves
using BSON

main_path = "/scratch/cmpe299-fa22/tristan/data/design_space=build_triple_ring_design_space_freq=1000.0"
data_path = joinpath(main_path, "episodes")

paths = filter(isdir, readdir(data_path, join = true))
paths = vcat(readdir.(paths, join = true)...)
paths = filter(x -> occursin(".bson", x), paths)
episode_length = length(EpisodeData(path = paths[1]))

dataset_path = mkpath(joinpath(main_path, "dataset/train"))

for (i, path) in enumerate(paths)

    episode = EpisodeData(path = path)
    episode_path = mkpath(joinpath(dataset_path, "episode$i"))

    for j in 1:length(episode)

        BSON.bson(
            joinpath(episode_path, "step_$j.bson"),
            state = episode.states[j],
            action = episode.actions[j],
            tspan = episode.tspans[j],
            sigma = episode.sigmas[j]
            )
    end
end