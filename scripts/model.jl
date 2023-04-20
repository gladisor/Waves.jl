include("dependencies.jl")

dim = TwoDim(grid_size, elements)
pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)

random_design_kwargs = Dict(:width => 1, :hight => 2, :spacing => 3.0f0, :r => Waves.MAX_RADII, :c => 2100.0f0, :center => [0.0f0, 0.0f0])
initial_design = random_radii_scatterer_formation(;random_design_kwargs...)

env = gpu(ScatteredWaveEnv(
    dim,
    initial_condition = pulse,
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

@save "results/radii/env.bson" env

policy = RandomDesignPolicy(action_space(env))
data = generate_episode_data(policy, env, 10)

for (i, episode) in enumerate(data)
    plot_episode_data!(episode, cols = 5, path = "results/radii/data$i.png")
    plot_sigma!(episode, path = "results/radii/sigma$i.png")
end


# data = generate_episode_data(policy, env, 100)

# latent_dim = OneDim(grid_size, 1024)
# latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 5000.0f0)

# activation = relu
# model = WaveControlModel(
#     WaveEncoder(6, 64, 1, tanh),
#     DesignEncoder(2 * length(vec(initial_design)), 256, 1024, activation),
#     Integrator(runge_kutta, latent_dynamics, ti, dt, steps),
#     Chain(Flux.flatten, Dense(3072, 1024, activation), Dense(1024, 1024, activation), Dense(1024, 1), vec)
#     ) |> gpu

# episode = data[1]
# s, d, a, sigma = episode.states[5].wave_total, episode.states[5].design, episode.actions[5], episode.sigmas[5]
# s, d, a, sigma = gpu(s), gpu(d), gpu(a), gpu(sigma)

# states = vcat([d.states for d in data]...)
# actions = vcat([d.actions for d in data]...)
# sigmas = vcat([d.sigmas for d in data]...)

# train_loader = Flux.DataLoader((states, actions, sigmas), shuffle = true)
# println("Train Loader Length: $(length(train_loader))")

# zi = encode(model, s, d, a)
# z = model.iter(zi)
# render!(latent_dim, cpu(z), path = "results/latent_wave_original.mp4")

# model = train(model, train_loader)

# plot_sigma!(model, episode, path = "results/sigma_opt.png")

# zi = encode(model, s, d, a)
# z = model.iter(zi)
# render!(latent_dim, cpu(z), path = "results/latent_wave_opt.mp4")

# validation_episode = generate_episode_data(policy, env, 10)
# for (i, ep) in enumerate(validation_episode)
#     plot_sigma!(model, ep, path = "results/validation_ep$i.png")
# end

# model = cpu(model)
# @save "results/model.bson" model
# env = cpu(env)
# @save "results/env.bson" env