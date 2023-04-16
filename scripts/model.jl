using CairoMakie
using Flux
using Flux: unbatch
using Interpolations
using Interpolations: Extrapolation
using Waves

include("plot.jl")
using Flux: Params, Recur
using Waves: speed
include("../src/dynamics.jl")
include("env.jl")

grid_size = 8.0f0
elements = 256
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100
tspan = build_tspan(ti, dt, steps)
tf = tspan[end]

dim = TwoDim(grid_size, elements)
grid = build_grid(dim)
grad = build_gradient(dim)
bc = dirichlet(dim)
pml = build_pml(dim, 2.0f0, 20000.0f0)
wave = build_wave(dim, fields = 6)

pulse = Pulse(dim, -5.0f0, 0.0f0, 1.0f0)
wave = pulse(wave)

initial = Scatterers([0.0f0 0.0f0], [1.0f0], [2100.0f0])
design = linear_interpolation([ti, tf], [initial, initial])

dynamics = SplitWavePMLDynamics(design, dim, grid, ambient_speed, grad, bc, pml)
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

env = ScatteredWaveEnv(
    wave, wave,
    SplitWavePMLDynamics(design, dim, grid, ambient_speed, grad, bc, pml),
    SplitWavePMLDynamics(nothing, dim, grid, ambient_speed, grad, bc, pml),
    zeros(Float32, steps + 1),
    0,
    dt,
    steps)

designs = []
sigma = []
ts = []
us = []
sols = []

for i in 1:10
    action = Scatterers([(-1) ^ (i + 1) * 1.0f0 (-1) ^ i * 1.0f0], [0.0f0], [0.0f0])
    @time sol = env(action)

    push!(sols, sol)
    push!(designs, env.total.design)
    push!(sigma, env.σ)
end

design_tspan = vcat(sols[1].t[1], [sol.t[end] for sol in sols]...)
ds = [designs[1](0.0f0), [d(sol.t[end]) for (d, sol) in zip(designs, sols)]...]
design_interp = linear_interpolation(design_tspan, ds)

sol = WaveSol(sols...)
u_interp = linear_interpolation(sol.t, sol.u)

render!(dim, sol.t, u_interp, design_interp, seconds = 5.0f0)

sigma = [sigma[1][1], [s[2:end] for s in sigma]...]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, sol.t, vcat(sigma...), color = :blue)
save("sigma2.png", fig)