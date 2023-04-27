include("dependencies.jl")

grid_size = 8.0f0
elements = 256
ambient_speed = AIR
pml_width = 2.0f0
pml_scale = 20000.0f0
dt = 0.00005f0
steps = 100
actions = 10
h_channels = 8
h_size = 512
latent_elements = 1024
latent_pml_width = 1.0f0
latent_pml_scale = 20000.0f0
n_mlp_layers = 2
horizon = 2
lr = 1e-4

episodes = 100
epochs = 10

dim = TwoDim(grid_size, elements)
pulse = Pulse(dim, x = -5.0f0, y = 0.0f0, intensity = 1.0f0)
random_radii = RandomRadiiScattererGrid(width = 1, height = 2, spacing = 3.0f0, c = BRASS, center = zeros(Float32, 2))
ds = radii_design_space(random_radii(), 1.0f0)

env = gpu(WaveEnv(
    dim,
    reset_wave = pulse,
    reset_design = random_radii,
    action_space = ds,
    sensor = DisplacementImage(),
    ambient_speed = ambient_speed,
    pml_width = pml_width,
    pml_scale = pml_scale,
    dt = dt,
    integration_steps = steps,
    actions = actions))

policy = RandomDesignPolicy(action_space(env))

reset!(env)
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

latent_dim = OneDim(grid_size, latent_elements)

render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_original.mp4"))
train_loader = Flux.DataLoader(prepare_data(data, horizon), shuffle = true)
model = train(model, train_loader, epochs, lr)
render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_opt.mp4"))

val_episodes = generate_episode_data(policy, env, 10)
for (i, episode) in enumerate(val_episodes)
    plot_sigma!(model, episode, path = joinpath(path, "val_episode$i.png"))
end

model = cpu(model)
env = cpu(env)
@save joinpath(path, "model.bson") model
@save joinpath(path, "env.bson") env
