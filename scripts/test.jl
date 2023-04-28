include("dependencies.jl")

dim = TwoDim(8.0f0, 256)
pulse = build_pulse(build_grid(dim), 0.0f0, -3.0f0, 5.0f0)

random_radii = RandomRadiiScattererGrid(width = 1, height = 2, spacing = 3.0f0, c = BRASS, center = zeros(Float32, 2))
ds = radii_design_space(random_radii(), 1.0f0)

env = WaveEnv(
    dim,
    reset_wave = Silence(),
    reset_design = random_radii,
    action_space = radii_design_space(random_radii(), 1.0f0),
    source = Source(pulse, freq = 300.0f0),
    sensor = DisplacementImage(),
    ambient_speed = AIR) |> gpu

reset!(env)

s = state(env)

# tspan = build_tspan(iter)
# @time u = iter(wave) |> cpu

# u = linear_interpolation(tspan, unbatch(u))
# @time render!(dim, tspan, u, path = "vid.mp4", seconds = 5.0f0)