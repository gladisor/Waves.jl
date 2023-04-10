using CairoMakie
using Flux
using Flux: batch, unbatch
using Waves

include("plot.jl")
include("env.jl")

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00002f0
steps = 100

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = TwoDim(grid_size, elements)
g = grid(dim)
grad = build_gradient(dim)
pml = build_pml(dim, 2.0f0, 50.0f0 * ambient_speed)

initial = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])
design = DesignInterpolator(initial)

pulse = Pulse(dim, -4.0f0, 0.0f0, pulse_intensity)
ui = pulse(build_wave(dim, fields = 6))

env = ScatteredWaveEnv(
    ui, ui,
    SplitWavePMLDynamics(design, dim, g, ambient_speed, grad, pml),
    SplitWavePMLDynamics(nothing, dim, g, ambient_speed, grad, pml),
    zeros(Float32, steps + 1), 0, dt, steps)

iter = Integrator(runge_kutta, env.incident, time(env), dt, steps)

for i in 1:10
    action = Scatterers([0.0f0 0.0f0], [-1.0f0^i * 0.2f0], [0.0f0])
    u = @time batch(env(action))
    display(size(u))
    plot_solution!(2, 2, dim, u, path = "u$i.png")
end