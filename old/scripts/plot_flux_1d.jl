using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = OneDim(size = gs)
wave = Wave(dim = dim)
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 100, :dt => 0.1)

C = WaveSpeed(wave)
x, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x) ^ 2
Dt = Differential(t); Dtt = Differential(t) ^ 2

@variables ∇(..), ∇²(..)

eq = [
    Waves.wave_equation(wave),
    ∇(x, t) ~ Dx(u(x, t)),
    ∇²(x, t) ~ Dxx(u(x, t))
    ]

bcs = [
    u(x, 0.0) ~ kwargs[:ic](wave),
    kwargs[:boundary](wave)...
    ]
    
ps = [wave.speed => kwargs[:ambient_speed]]

@named sys = PDESystem(
    eq, bcs, Waves.get_domain(wave, tmax = kwargs[:tmax]), Waves.spacetime(wave), 
    [
        Waves.signature(wave),
        ∇(x, t),
        ∇²(x, t),
    ], ps)

disc = Waves.wave_discretizer(wave, kwargs[:n])
@time iter = init(discretize(sys, disc), Tsit5(), advance_to_tstop = true, saveat = kwargs[:dt])
@time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])

propagate!(sim)
displacement = Waves.extract(sim, sim.grid[u(x, t)])
gradient = Waves.extract(sim, sim.grid[∇(x, t)])
laplacian = Waves.extract(sim, sim.grid[∇²(x, t)])

line_integral_flux = [gradient[i][[1, end]]' * [-1.0, 1.0] for i ∈ axes(gradient, 1)]
surface_integral_flux = sum.(laplacian) ./ 10.0

import GLMakie

sol = WaveSol(sim)
tick_length = length(line_integral_flux)
old_ticks = collect(1:100:tick_length)
new_ticks = collect(range(0, sol.tspan[end], length = length(old_ticks)))

fig = GLMakie.Figure(fontsize = 20)
ax = GLMakie.Axis(fig[1,1], 
    title = "Flux on the boundary of simulation", 
    xlabel = "Time", ylabel = "Flux", xticks = (old_ticks,  string.(new_ticks)))

GLMakie.lines!(ax, line_integral_flux, label = "Line Integral", linewidth = 2)
GLMakie.lines!(ax, surface_integral_flux, label = "Surface Integral", linewidth = 2)

GLMakie.Legend(fig[1, 2], ax, "Method")
GLMakie.save("comparison.png", fig)