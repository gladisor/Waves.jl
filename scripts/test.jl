include("dependencies.jl")

initial_design = random_radii_scatterer_formation(;random_design_kwargs...)

dim = TwoDim(grid_size, elements)
env = gpu(ScatteredWaveEnv(
    dim,
    initial_condition = Pulse(dim, -5.0f0, 0.0f0, 1.0f0),
    design = initial_design,
    ambient_speed = ambient_speed,
    pml_width = pml_width,
    pml_scale = pml_scale,
    reset_design = d -> gpu(random_radii_scatterer_formation(;random_design_kwargs...)),
    action_space = radii_design_space(initial_design, 1.0f0),
    dt = dt,
    integration_steps = steps,
    actions = 10
))

policy = RandomDesignPolicy(action_space(env))

reset!(env)
s = state(env)
a = policy(env)

wave_encoder = Chain(
    WaveEncoder(6, 8, 2, tanh), 
    Dense(1024, latent_elements, tanh),
    z -> hcat(z[:, 1], z[:, 2] * 0.0f0))

design_encoder = Chain(
    Dense(2 * length(vec(s.design)), h_size, relu), 
    Dense(h_size, 2 * latent_elements),
    z -> reshape(z, latent_elements, :),
    z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2])))

latent_dim = OneDim(grid_size, latent_elements)
grad = build_gradient(latent_dim)
pml = rand(Float32, size(latent_dim, 1))
bc = dirichlet(latent_dim)

latent_dynamics = ForceLatentDynamics(ambient_speed, latent_pml_scale, grad, pml, bc)
iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)
mlp = Chain(flatten, Dense(latent_elements * 4, h_size, relu), Dense(h_size, 1), vec)
model = gpu(WaveMPC(wave_encoder, design_encoder, iter, mlp))

data = generate_episode_data(policy, env, 50)

plot_sigma!(model, data[1], path = "episode_sigma_original.png")
render_latent_wave!(latent_dim, model, s, a, path = "latent_wave_original.mp4")

train_loader = Flux.DataLoader(prepare_data(data, 1), shuffle = true)
model = train(model, train_loader, 15)

plot_sigma!(model, data[1], path = "episode_sigma_opt.png")
render_latent_wave!(latent_dim, model, s, a, path = "latent_wave_original.mp4")

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(pml))
lines!(ax, cpu(model.iter.dynamics.pml))
save("pml.png", fig)

model = cpu(model)
env = cpu(env)

@save "model.bson" model
@save "env.bson" env