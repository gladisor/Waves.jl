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
    actions = 10))

policy = RandomDesignPolicy(action_space(env))

episodes = 100

p = Progress(episodes)

for i in 1:episodes
    episode = generate_episode_data(policy, env)
    @save "data/episode$i.bson" episode
    next!(p)
end