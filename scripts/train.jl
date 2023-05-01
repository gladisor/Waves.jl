include("dependencies.jl")

grid_size = 15.0f0
elements = 512
ambient_speed = AIR
pml_width = 2.0f0
pml_scale = 20000.0f0
dt = 0.00005f0
steps = 100
actions = 100
h_channels = 16
h_size = 512
latent_elements = 1024
latent_pml_width = 1.0f0
latent_pml_scale = 20000.0f0
n_mlp_layers = 2
horizon = 10
lr = 1e-4
epochs = 10

env = gpu(BSON.load("data/env.bson")[:env])

reset!(env)
policy = RandomDesignPolicy(action_space(env))

s = gpu(state(env))
a = gpu(policy(env))

model = gpu(build_wave_control_model(
    in_channels = 1,
    h_channels = h_channels,
    design_size = length(vec(s.design)),
    action_size = length(vec(a)),
    h_size = h_size,
    latent_grid_size = grid_size,
    latent_elements = latent_elements,
    latent_pml_width = latent_pml_width,
    latent_pml_scale = latent_pml_scale,
    ambient_speed = ambient_speed,
    dt = dt,
    steps = steps,
    n_mlp_layers = n_mlp_layers,
))

data = Vector{EpisodeData}([
    EpisodeData(path = "data/episode1/episode.bson"),
    EpisodeData(path = "data/episode2/episode.bson"),
    EpisodeData(path = "data/episode3/episode.bson"),
    EpisodeData(path = "data/episode4/episode.bson"),
    EpisodeData(path = "data/episode5/episode.bson"),
    EpisodeData(path = "data/episode6/episode.bson"),
    EpisodeData(path = "data/episode7/episode.bson"),
    EpisodeData(path = "data/episode8/episode.bson"),
    EpisodeData(path = "data/episode9/episode.bson"),
    EpisodeData(path = "data/episode10/episode.bson"),
    ])

path = mkpath("results/cloak_h_channels=$(h_channels)_h_size=$(h_size)_latent_elements=$(latent_elements)_n_mlp_layers=$(n_mlp_layers)_horizon=$(horizon)_lr=$(lr)")

latent_dim = OneDim(grid_size, latent_elements)
render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_original.mp4"))

train_loader = Flux.DataLoader(prepare_data(data, horizon), shuffle = true)
model = train(model, train_loader, epochs, lr)

BSON.bson(joinpath(path, "model.bson"), model = cpu(model))

render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_opt.mp4"))

val_episodes = generate_episode_data(policy, env, 1)
for (i, episode) in enumerate(val_episodes)
    plot_sigma!(model, episode, path = joinpath(path, "val_episode$i.png"))
end