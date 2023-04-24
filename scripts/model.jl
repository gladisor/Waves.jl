include("dependencies.jl")

path = "results/radii/WaveMPC"

initial_design = random_radii_scatterer_formation(;random_design_kwargs...)

dim = TwoDim(grid_size, elements)
env = gpu(ScatteredWaveEnv(
    dim,
    initial_condition = Pulse(dim, -5.0f0, 0.0f0, 1.0f0),
    design = initial_design,
    ambient_speed = ambient_speed,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    reset_design = d -> gpu(random_radii_scatterer_formation(;random_design_kwargs...)),
    action_space = radii_design_space(initial_design, 1.0f0),
    dt = dt,
    integration_steps = steps,
    actions = 10
))

reset!(env)

policy = RandomDesignPolicy(action_space(env))
;

data = generate_episode_data(policy, env, 100)
episode = first(data)
plot_episode_data!(episode, cols = 5, path = joinpath(path, "data.png"))
plot_sigma!(episode, path = joinpath(path, "episode_sigma.png"))

idx = 6
s = gpu(episode.states[idx])
a = gpu(episode.actions[idx])
sigma = gpu(episode.sigmas[idx])

wave = s.wave_total
design = s.design
design_size = length(vec(design))

h_size = 128
latent_elements = 512
latent_dim = OneDim(grid_size, latent_elements)

latent_dynamics = ForceLatentDynamics(ambient_speed, build_gradient(latent_dim), dirichlet(latent_dim))
model = gpu(WaveMPC(
    Chain(WaveEncoder(6, 8, 2, tanh), Dense(1024, latent_elements, tanh)),
    Chain(Dense(2 * design_size, h_size, relu), Dense(h_size, 2 * latent_elements)),
    Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
    Chain(flatten, Dense(latent_elements * 4, h_size, relu), Dense(h_size, 1), vec)
))

## package data into train_loader
states = vcat([d.states for d in data]...)
actions = vcat([d.actions for d in data]...)
sigmas = vcat([d.sigmas for d in data]...)
train_loader = Flux.DataLoader((states, actions, sigmas), shuffle = true)
println("Train Loader Length: $(length(train_loader))")

# plot the latent wave before training
plot_action_distribution!(model, policy, env, path = joinpath(path, "action_distribution_original.png"))
render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_original.mp4"))

# ## train the model
model = train(model, train_loader, 20)
## plot latent wave after training
plot_action_distribution!(model, policy, env, path = joinpath(path, "action_distribution_opt.png"))
render_latent_wave!(latent_dim, model, s, a, path = joinpath(path, "latent_wave_opt.mp4"))

## generate and plot prediction performance after training
validation_episode = generate_episode_data(policy, env, 2)
for (i, ep) in enumerate(validation_episode)
    plot_sigma!(model, ep, path = joinpath(path, "validation_ep$i.png"))
end

## saving model and env
model = cpu(model)
@save joinpath(path, "model.bson") model
env = cpu(env)
@save joinpath(path, "env.bson") env