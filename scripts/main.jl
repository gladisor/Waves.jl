using Waves
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

design = Design(Configuration([-2.0, 0.0, 2.0], [-3.0, -3.0, -3.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]))
new_design = Configuration([-2.0, 0.0, 2.0], [3.0, 3.0, 3.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), 
    :boundary => OpenBoundary(), 
    :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.05)

println("Build sim")
@time sim = WaveSim(;design = design, kwargs...)
dp = vcat(design_parameters(design), design_parameters(new_design))
Waves.set_design_params!(sim, dp)

@time sim_inc = WaveSim(;kwargs...)

propagate!(sim)
propagate!(sim_inc)

sol = WaveSol(sim)
sol_inc = WaveSol(sim_inc)
sol_sc = sol - sol_inc

render!(sol_sc, design = range(design.design, new_design, length(sol)), path = "2d.mp4")

# C = WaveSpeed(wave, design)
# eq = Waves.wave_equation(wave, C)
# ps = [wave.speed => kwargs[:ambient_speed]]
# ps = vcat(ps, Waves.design_parameters(design, new_design, 0.0, kwargs[:tmax]))
# bcs = [wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave), kwargs[:boundary](wave)...]

# println("Build sys"); @time @named sys = PDESystem(eq, bcs, Waves.get_domain(wave, tmax = kwargs[:tmax]), Waves.spacetime(wave), [Waves.signature(wave)], ps)
# disc = Waves.wave_discretizer(wave, kwargs[:n])
# println("Build iter"); @time iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
# println("Build sim"); @time sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
# println("propagate!"); @time propagate!(sim)