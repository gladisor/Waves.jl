using Waves
using Waves: perturb

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 40.0, :n => 21, :dt => 0.05)
design = ParameterizedDesign(Cylinder(0.0, 0.0, 1.0, 0.00))
sim_tot = WaveSim(design = design; kwargs...)
sim_inc = WaveSim(;kwargs...)

propagate!(sim_inc)
sol_inc = WaveSol(sim_inc)

env = WaveEnv(sim = sim_tot, design = design, design_steps = 20)
design_trajectory = Vector{typeof(env.design.design)}([env.design.design])

while !is_terminated(env)
    action = Cylinder(env.sim.wave.dim, r = 0.0, c = 0.0)
    action = Cylinder(action.x/2, action.y/2, action.r, action.c)
    steps = perturb(env, action)

    [push!(design_trajectory, s) for s in steps]
end

sol_tot = WaveSol(env)
render!(sol_tot, design = design_trajectory, path = "vid_tot.mp4")

sol_sc = sol_tot - sol_inc
render!(sol_sc, design = design_trajectory, path = "vid_sc.mp4")

Waves.plot_energy!(sol_inc = sol_inc, sol_sc = sol_sc, path = "energy.png")
