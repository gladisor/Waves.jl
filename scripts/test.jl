include("dependencies.jl")

dim = TwoDim(8.0f0, 256)
pulse = build_pulse(build_grid(dim), 0.0f0, 0.0f0, 10.0f0)

env = WaveEnv(
    dim, 
    reset_wave = Silence(),
    reset_design = () -> NoDesign(),
    action_space = NoDesign()..NoDesign(),
    source = Source(pulse, freq = 200.0f0),
    actions = 20)

@time render!(policy, env, path = "vid.mp4", seconds = env.actions * 1.0f0)