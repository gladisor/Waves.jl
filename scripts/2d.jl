using CairoMakie
using Flux
using Waves

include("plot.jl")

grid_size = 10.f0
elements = 256
ti = 0.0f0
dt = 0.00002f0
steps = 200

ambient_speed = 1531.0f0
pulse_intensity = 1.0f0

dim = TwoDim(grid_size, elements)
g = grid(dim)
grad = build_gradient(dim)
pml = build_pml(dim, 2.0f0, 50.0f0 * ambient_speed)

initial = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])
# action = Scatterers([0.0f0 0.0f0], [0.2f0], [0.0f0])
# tf = iter.ti + iter.steps*iter.dt
# design = DesignInterpolator(initial, action, iter.ti, tf)
design = DesignInterpolator(initial)

dynamics = SplitWavePMLDynamics(design, dim, g, ambient_speed, grad, pml)
iter = Integrator(runge_kutta, dynamics, 0.0f0, dt, steps) |> gpu

pulse = Pulse(dim, -4.0f0, 0.0f0, pulse_intensity)
ui = build_wave(dim, fields = 6)

ui = pulse(ui) |> gpu

for i in 1:10
    @time u = iter(ui)
    plot_solution!(2, 2, cpu(dim), cpu(u), path = "u$i.png")
    ui = u[:, :, :, end]
end