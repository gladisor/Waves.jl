using Wandb

lg = WandbLogger(project = "Waves.jl")
wa = WandbArtifact("full_state_triple_ring_dataset", type = "dataset")

paths = readdir("/scratch/cmpe299-fa22/tristan/data/full_state_triple_ring_dataset/episodes/", join = true)
episode_paths = filter(isdir, paths)

for path in episode_paths
    Wandb.add_file(wa, joinpath(path, "episode.bson"), name = path)
end

Wandb.log(lg, wa)
close(lg)