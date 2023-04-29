include("dependencies.jl")

dim = TwoDim(20.0f0, 512)
pulse = build_pulse(build_grid(dim), -10.0f0, 0.0f0, 10.0f0)


random_radii = RandomRadiiScattererGrid(
    width = 3, height = 5, spacing = 3.0f0, 
    c = BRASS, center = [0.0f0, 0.0f0])

ds = radii_design_space(random_radii(), 1.0f0)

env = WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = random_radii,
    action_space = ds,
    source = Source(pulse, freq = 300.0f0),
    actions = 10) |> gpu

policy = RandomDesignPolicy(action_space(env))

@time render!(policy, env, path = "vid.mp4", seconds = env.actions * 1.0f0)