using Flux
using Flux: Recur, unbatch
using ReinforcementLearning
using Interpolations
using Interpolations: Extrapolation
using CairoMakie

using FileIO
using Waves

Flux.device!(1)

dim = TwoDim(15.0f0, 512)
grid = build_grid(dim)

pos = [0.0f0 0.0f0]

r_low = fill(0.2f0, size(pos, 1))
r_high = fill(1.0f0, size(pos, 1))
c = fill(Waves.AIR, size(pos, 1))

core = Cylinders([5.0f0, 0.0f0]', [2.0f0], [AIR])

design_low = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_low, c)), core)
design_high = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_high, c)), core)

design_space = DesignSpace(design_low, design_high)
include("plot.jl")

env = WaveEnv(
    dim,
    reset_wave = Silence(),
    design_space = DesignSpace(design_low, design_high),
    action_speed = 500.0f0,
    source = Source(build_pulse(grid, -10.0f0, 0.0f0, 10.0f0), freq = 2000.0f0),
    sensor = DisplacementImage(),
    ambient_speed = WATER,
    pml_width = 5.0f0,
    pml_scale = 10000.0f0,
    dt = Float32(1e-5),
    integration_steps = 100,
    actions = 50) |> gpu

policy = RandomDesignPolicy(action_space(env))

@time episode = generate_episode_data(policy, env)

plot_sigma!(episode, path = "sigma.png")
@time render!(policy, env, path = "vid.mp4", seconds = env.actions * 0.5f0)