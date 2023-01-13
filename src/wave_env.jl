export WaveEnv, state, is_terminated

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
    env.sim.iter.p[2:end-2] .= vcat(design_parameters(env.design), design_parameters(new_design))
    env.design.design = new_design
    Waves.step!(env.sim)
    return steps
end

function is_terminated(env::WaveEnv)
    return env.sim.iter.t >= env.sim.prob.tspan[end]
end

function Base.display(env::WaveEnv)
    println(typeof(env))
end