include("dependencies.jl")

dim = TwoDim(15.0f0, 512)
grid = build_grid(dim)

ring = RandomRadiiScattererRing(5.0f0, 1.0f0, 4, BRASS, [0.0f0, 0.0f0])
core = Scatterers([0.0f0 0.0f0], [1.6f0], [BRASS])

cloak = RandomCloak(ring, core)
pulse = build_pulse(grid, -12.0f0, 0.0f0, 10.0f0)

env = WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = cloak,
    action_space = design_space(cloak, 1.0f0),
    source = Source(pulse, freq = 300.0f0),
    sensor = DisplacementImage(),
    ambient_speed = AIR,
    actions = 100) |> gpu

policy = RandomDesignPolicy(action_space(env))

for i in 1:10
    path = mkpath("data/episode$i")
    @time episode = generate_episode_data(policy, env)
    save(episode, joinpath(path, "episode.bson"))
    plot_sigma!(episode, path = joinpath(path, "sigma.png"))
end

BSON.bson("data/env.bson", env = cpu(env))
@time render!(policy, env, path = "data/vid.mp4", seconds = env.actions * 1.0f0)