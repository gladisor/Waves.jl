include("dependencies.jl")

grid_size = 15.0f0
dim = TwoDim(grid_size, 512)
grid = build_grid(dim)

offset = [4.f0 0.0f0]
pos = hexagon(3.0f0) .+ offset
config = Radii(pos, BRASS)
core = Scatterers(offset, [1.0f0], [BRASS])
cloak = RandomCloak(config, core)
pulse = build_pulse(grid, -10.0f0, 0.0f0, 5.0f0)

action_scale = 0.25f0
pml_width = 2.0f0
pml_scale = 10000.0f0
actions = 100
integration_steps = 100

println("Building WaveEnv")
env = gpu(WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = cloak,
    action_space = design_space(cloak, action_scale),
    source = Source(pulse, freq = 200.0f0),
    sensor = DisplacementImage(),
    ambient_speed = AIR,
    pml_width = pml_width,
    pml_scale = pml_scale,
    actions = actions,
    integration_steps = integration_steps,
    dt = Float32(5e-5)))

policy = RandomDesignPolicy(action_space(env))
data_path = mkpath("data/hexagon_large_grid")

println("Rendering Example")
@time render!(policy, env, path = joinpath(data_path, "vid.mp4"), seconds = env.actions * 0.5f0)

println("Generating Data")
for i in 1:200
    path = mkpath(joinpath(data_path, "episode$i"))
    @time episode = generate_episode_data(policy, env)
    save(episode, joinpath(path, "episode.bson"))
    plot_sigma!(episode, path = joinpath(path, "sigma.png"))
end

env = cpu(env)
BSON.bson(joinpath(data_path, "env.bson"), env = env)
