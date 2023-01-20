export WaveEnv, state, is_terminated, perturb

Base.Base.@kwdef mutable struct WaveEnv{Dm <: AbstractDim, Dn <: AbstractDesign}
    sim::WaveSim{Dm}
    design::ParameterizedDesign{Dn}
    design_steps::Int
end

function reset!(env::WaveEnv)
    reset!(env.sim)
end

function state(env::WaveEnv)
    return state(env.sim)
end

function perturb(env::WaveEnv, action::AbstractDesign)
    t0 = env.sim.iter.t
    tf = t0 + env.design_steps * env.sim.dt

    set_t0!(env.sim, t0)
    set_tf!(env.sim, tf)
    add_tstop!(env.sim.iter, tf)

    new_design = env.design + action
    steps = range(env.design.design, new_design, env.design_steps)
    dp = vcat(design_parameters(env.design), design_parameters(new_design))
    set_design_params!(env.sim, dp)
    
    env.design.design = new_design
    Waves.step!(env.sim)
    return steps
end

function is_terminated(env::WaveEnv)
    return current_time(env.sim) >= get_tmax(env.sim)
end

function Base.display(env::WaveEnv)
    println(typeof(env))
end

function WaveSol(env::WaveEnv)
    return WaveSol(env.sim)
end