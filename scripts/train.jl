include("dependencies.jl")

grid_size = 8.0f0
elements = 256
ambient_speed = AIR
pml_width = 2.0f0
pml_scale = 20000.0f0
dt = 0.00005f0
steps = 100
actions = 20
h_channels = 16
h_size = 512
latent_elements = 1024
latent_pml_width = 1.0f0
latent_pml_scale = 20000.0f0
n_mlp_layers = 2
horizon = 3
lr = 1e-4
episodes = 50
epochs = 10

dim = TwoDim(grid_size, elements)
pulse = build_pulse(build_grid(dim), -5.0f0, 0.0f0, 5.0f0)

random_radii = RandomRadiiScattererGrid(
    width = 1, height = 2, spacing = 3.0f0, 
    c = BRASS, center = zeros(Float32, 2))

ds = radii_design_space(random_radii(), 1.0f0)

env = WaveEnv(
    dim,
    reset_wave = Silence(),
    reset_design = random_radii,
    action_space = radii_design_space(random_radii(), 1.0f0),
    source = Source(pulse, freq = 300.0f0),
    sensor = DisplacementImage(),
    ambient_speed = AIR,
    actions = actions) |> gpu

reset!(env)
policy = RandomDesignPolicy(action_space(env))

s = gpu(state(env))
a = gpu(policy(env))

model = gpu(build_wave_control_model(
    in_channels = 1,
    h_channels = h_channels,
    design_size = length(vec(s.design)),
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

data = generate_episode_data(policy, env, episodes)

path = mkpath("results/h_channels=$(h_channels)_h_size=$(h_size)_latent_elements=$(latent_elements)_n_mlp_layers=$(n_mlp_layers)_horizon=$(horizon)_lr=$(lr)")

plot_episode_data!(data[1], cols = 5, path = joinpath(path, "episode_data.png"))

latent_dim = OneDim(grid_size, latent_elements)
render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_original.mp4"))
train_loader = Flux.DataLoader(prepare_data(data, horizon), shuffle = true)
model = train(model, train_loader, epochs, lr)
render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_opt.mp4"))

val_episodes = generate_episode_data(policy, env, 10)
for (i, episode) in enumerate(val_episodes)
    plot_sigma!(model, episode, path = joinpath(path, "val_episode$i.png"))
    plot_episode_data!(episode, cols = 5, path = joinpath(path, "val_episode_data$i.png"))
end

model = cpu(model)
env = cpu(env)
@save joinpath(path, "model.bson") model
@save joinpath(path, "env.bson") env