using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using Distributions: Uniform
import GLMakie

include("configuration.jl")

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

M = 4

design = Design(Configuration(
    [-2.0, -2.0, -2.0, -4.0], 
    [-3.0, 0.0, 3.0, 3.0], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.1, 0.1, 0.1, 0.1]), M = M)

new_design = Configuration(
    [2.0, 2.0, 2.0, 4.0], 
    [-3.0, 0.0, 3.0, 3.0], 
    [0.5, 0.5, 0.5, 0.5], 
    [0.1, 0.1, 0.1, 0.1])

C = WaveSpeed(wave, design)

x, y, t, u = Waves.unpack(wave)
Dx = Differential(x); Dxx = Differential(x)^2; Dy = Differential(y); Dyy = Differential(y)^2
Dt = Differential(t); Dtt = Differential(t)^2

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), :boundary => MinimalBoundary(), 
    :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.05)

ps = [wave.speed => kwargs[:ambient_speed]]

C = WaveSpeed(wave, design)
eq = Dtt(u(x, y, t)) ~ C(x, y, t) ^ 2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))
ps = vcat(ps, Waves.design_parameters(design, new_design, 0.0, kwargs[:tmax]))

bcs = [wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), kwargs[:boundary](wave)...]

println("Build sys"); @time @named sys = PDESystem(eq, bcs, Waves.get_domain(wave, tmax = kwargs[:tmax]), Waves.spacetime(wave), [Waves.signature(wave)], ps)
disc = Waves.wave_discretizer(wave, kwargs[:n])
println("Build iter"); @time iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
println("Build sim"); @time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
println("propagate!"); @time propagate!(sim)

sol = WaveSol(sim)
steps = range(design.design, new_design, length(sol))
println("render!"); @time render!(sol, design = steps, path = "2d.mp4")