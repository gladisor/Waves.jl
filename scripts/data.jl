include("dependencies.jl")

dim = TwoDim(8.0f0, 256)
grid = build_grid(dim)

# config = RandomRadiiScattererGrid(width = 1, height = 2, spacing = 0.1f0, c = BRASS, center = [0.0f0, 0.0f0])
cloak = RandomRadiiScattererGrid(width = 1, height = 2, spacing = 0.1f0, c = BRASS, center = [0.0f0, 0.0f0])

# core = Scatterers([5.0f0 0.0f0], [1.6f0], [BRASS])
# cloak = RandomCloak(config, core)
pulse = build_pulse(grid, -6.0f0, 0.0f0, 10.0f0)

action_scale = 1.0f0
actions = 50
integration_steps = 100

env = gpu(WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = cloak,
    action_space = design_space(cloak, action_scale),
    source = Source(pulse, freq = 300.0f0),
    sensor = DisplacementImage(),
    ambient_speed = AIR,
    actions = actions,
    integration_steps = integration_steps))

policy = RandomDesignPolicy(action_space(env))
data_path = mkpath("data/M=2")

@time render!(policy, env, path = joinpath(data_path, "vid.mp4"), seconds = env.actions * 0.5f0)

for i in 1:20
    path = mkpath(joinpath(data_path, "episode$i"))
    @time episode = generate_episode_data(policy, env)
    save(episode, joinpath(path, "episode.bson"))
    plot_sigma!(episode, path = joinpath(path, "sigma.png"))
end

env = cpu(env)
BSON.bson(joinpath(data_path, "env.bson"), env = env)
