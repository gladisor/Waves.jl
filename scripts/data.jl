include("dependencies.jl")

dim = TwoDim(8.0f0, 256)
grid = build_grid(dim)

config = Radii(hexagon(3.0f0) .+ [2.0f0 0.0f0], BRASS)
core = Scatterers([2.0f0 0.0f0], [1.0f0], [BRASS])
cloak = RandomCloak(config, core)

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
    integration_steps = integration_steps,
    dt = Float32(5e-5)))

policy = RandomDesignPolicy(action_space(env))
data_path = mkpath("data/M=6_as=$(action_scale)_additional_data")

@time render!(policy, env, path = joinpath(data_path, "vid.mp4"), seconds = env.actions * 0.5f0)

for i in 1:50
    path = mkpath(joinpath(data_path, "episode$i"))
    @time episode = generate_episode_data(policy, env)
    save(episode, joinpath(path, "episode.bson"))
    plot_sigma!(episode, path = joinpath(path, "sigma.png"))
end

env = cpu(env)
BSON.bson(joinpath(data_path, "env.bson"), env = env)
