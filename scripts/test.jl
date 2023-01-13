using Waves

grid_size = 5.0
design = ParameterizedDesign(Cylinder(-3.0, -3.0, 0.5, 0.0))

sim = WaveSim(
    wave = Wave(dim = TwoDim(-grid_size, grid_size, -grid_size, grid_size)),
    design = design, 
    ic = GaussianPulse(intensity = 5.0, loc = [2.5, 2.5]),
    t_max = 10.0,
    speed = 1.0, 
    n = 30, 
    dt = 0.05)

@time env = WaveEnv(sim, design, 40)

reset!(env)
env.design = ParameterizedDesign(Cylinder(0.0, 0.0, 0.5, 0.2))
steps = Vector{typeof(env.design.design)}([env.design.design])

@time while !is_terminated(env)
    action = Cylinder(randn(), randn(), 0.0, 0.0)
    [push!(steps, s) for s ∈ step(env, action)]
end

sol = WaveSol(env.sim)
steps = vcat(steps...)
@time render!(sol, design = steps, path = "env.gif")