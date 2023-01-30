using Waves
using Waves: perturb
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

design = Design(Configuration([0.0], [-3.0], [0.5], [0.0]))
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.05)

println("Build sim")
@time sim = WaveSim(;design = design, kwargs...)

env = WaveEnv(sim, design, 20)

steps = Vector{Configuration}([env.design.design])

@time while !is_terminated(env)
    action = Configuration(dim, M = length(env.design.design.x), r = 0.0) / 5
    [push!(steps, s) for s ∈ perturb(env, action)]
end

sol = WaveSol(env)
render!(sol, design = steps, path = "env_tot.mp4")

@time sim_inc = WaveSim(;kwargs...)
propagate!(sim_inc)
sol_inc = WaveSol(sim_inc)
sol_sc = sol - sol_inc
render!(sol_sc, design = steps, path = "env_sc.mp4")