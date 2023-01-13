using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using OrdinaryDiffEq: ODEIntegrator, ODESolution
using Waves

mutable struct WaveEnv{Dm <: AbstractDim, Dn <: AbstractDesign}
    sim::WaveSim{Dm}
    design::ParameterizedDesign{Dn}
    design_steps::Int
end

function Waves.reset!(env::WaveEnv)
    reset!(env.sim)
end

function state(env::WaveEnv)
    return env.sim.iter[env.sim.grid[Waves.signature(env.sim.wave)]]
end

function Base.step(env::WaveEnv, action::AbstractDesign)
    t0 = env.sim.iter.t
    tf = t0 + env.design_steps * env.sim.dt
    env.sim.iter.p[end-1] = t0
    env.sim.iter.p[end] = tf
    add_tstop!(env.sim.iter, tf)

    new_design = env.design + action
    steps = range(env.design.design, new_design, env.design_steps)
    sim.iter.p[2:end-2] .= vcat(design_parameters(env.design), design_parameters(new_design))
    env.design.design = new_design
    Waves.step!(env.sim)
    return steps
end

function is_terminated(env::WaveEnv)
    return env.sim.iter.t >= env.sim.prob.tspan[end]
end

grid_size = 5.0

cylinder = ParameterizedDesign(Cylinder(-3.0, -3.0, 0.5, 0.2))
@time sim = WaveSim(
    wave = Wave(dim = TwoDim(-grid_size, grid_size, -grid_size, grid_size)),
    design = cylinder, ic = GaussianPulse(intensity = 5.0, loc = [2.5, 2.5]),
    t_max = 20.0, speed = 1.0, n = 30, dt = 0.05)
;
@time env = WaveEnv(sim, cylinder, 10)
;
reset!(env)
env.design = ParameterizedDesign(Cylinder(0.0, 0.0, 0.5, 0.2))
steps = Vector{typeof(env.design.design)}([env.design.design])
;
@time while !is_terminated(env)
    action = Cylinder(randn() * 0.2, randn() * 0.2, 0.0, 0.0)
    [push!(steps, s) for s ∈ step(env, action)]
end
;
sol = WaveSol(env.sim)
steps = vcat(steps...)
;
@time render!(
    sol, 
    design = steps,
    path = "env.mp4")
;