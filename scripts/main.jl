using Waves
using Waves: perturb

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)
kwargs = Dict(:wave => wave, :ic => Silence(), :boundary => PlaneWave(), :ambient_speed => 2.0, :tmax => 10.0, :n => 21, :dt => 0.05)
design = ParameterizedDesign(Cylinder(0.0, 0.0, 1.0, 0.0))
sim_tot = WaveSim(design = design; kwargs...)
env = WaveEnv(sim = sim_tot, design = design, design_steps = 20)

design_trajectory = Vector{typeof(env.design.design)}([env.design.design])

while !is_terminated(env)
    action = Cylinder(env.sim.wave.dim, r = 0.0, c = 0.0)
    action = Cylinder(action.x/5, action.y/5, action.r, action.c)
    steps = perturb(env, action)

    [push!(design_trajectory, s) for s in steps]
end

render!(WaveSol(env), design = design_trajectory, path = "vid.mp4")

# x, y, t, u = unpack(wave)

# ∇u(x, y, t) ~ Dx(u(x, y, t)) + Dy(u(x, y, t))