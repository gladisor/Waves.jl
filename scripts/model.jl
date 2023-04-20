include("dependencies.jl")

path = "results/radii/PercentageWaveControlModel"

random_design_kwargs = Dict(:width => 1, :hight => 2, :spacing => 3.0f0, :r => Waves.MAX_RADII, :c => 2100.0f0, :center => [0.0f0, 0.0f0])
initial_design = random_radii_scatterer_formation(;random_design_kwargs...)

dim = TwoDim(grid_size, elements)
env = gpu(ScatteredWaveEnv(
    dim,
    initial_condition = Pulse(dim, -5.0f0, 0.0f0, 1.0f0),
    design = initial_design,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    reset_design = d -> gpu(random_radii_scatterer_formation(;random_design_kwargs...)),
    action_space = radii_design_space(initial_design, 1.0f0),
    dt = dt,
    integration_steps = steps,
    actions = 10
))
;

# policy = RandomDesignPolicy(action_space(env))
# data = generate_episode_data(policy, env, 3)

# for (i, episode) in enumerate(data)
#     plot_episode_data!(episode, cols = 5, path = joinpath(path, "data$i.png"))
#     plot_sigma!(episode, path = joinpath(path, "sigma$i.png"))
# end

latent_elements = 1024
latent_dim = OneDim(grid_size, latent_elements)
latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 5000.0f0)
design_size = 2 * length(vec(rand(action_space(env))))
model = PercentageWaveControlModel(
    WaveEncoder(6, 64, 1, tanh),
    Chain(Dense(latent_elements, latent_elements, tanh), vec),
    DesignEncoder(design_size, 512, latent_elements, relu),
    Chain(c -> 1.0f0 .+ (c .- 0.5f0) * 0.5f0),
    Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
    Chain(Flux.flatten, Dense(3 * latent_elements, latent_elements, relu), Dense(latent_elements, latent_elements, relu), Dense(latent_elements, 1), vec)) |> gpu

data = generate_episode_data(policy, env, 20)

episode = data[1]
s, d, a, sigma = episode.states[5].wave_total, episode.states[5].design, episode.actions[5], episode.sigmas[5]
s, d, a, sigma = gpu(s), gpu(d), gpu(a), gpu(sigma)

## package data into train_loader
states = vcat([d.states for d in data]...)
actions = vcat([d.actions for d in data]...)
sigmas = vcat([d.sigmas for d in data]...)
train_loader = Flux.DataLoader((states, actions, sigmas), shuffle = true)
println("Train Loader Length: $(length(train_loader))")

## plot the latent wave before training
z = model.iter(encode(model, s, d, a))
render!(latent_dim, cpu(z), path = joinpath(path, "latent_wave_original.mp4"))

## train the model
model = train(model, train_loader, 50)

## plot latent wave after training
z = model.iter(encode(model, s, d, a))
render!(latent_dim, cpu(z), path = joinpath(path, "latent_wave_opt.mp4"))

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