export WaveEnv, reset!, state, is_terminated

Base.@kwdef mutable struct WaveEnv{Dim <: AbstractDim, Design <: AbstractDesign}
    sim::WaveSim{Dim}
    design::ParameterizedDesign{Design}
    design_steps::Int
end

function reset!(env::WaveEnv)
    reset!(env.sim)
    env.design.design = typeof(env.design.design)(env.sim.wave.dim)
    return nothing
end

function state(env::WaveEnv)
    return state(env.sim)
end

function update_design!(env::WaveEnv, new_design::AbstractDesign)
    dp = vcat(design_parameters(env.design), design_parameters(new_design))
    set_design_params!(env.sim, dp)
    env.design.design = new_design
    return nothing
end

function perturb(env::WaveEnv, action::AbstractDesign)
    t0 = current_time(env.sim)
    tf = t0 + env.design_steps * env.sim.dt

    set_t0!(env.sim, t0)
    set_tf!(env.sim, tf)
    add_tstop!(env.sim.iter, tf)

    new_design = perturb(env.design.design, action, env.sim.wave.dim)
    steps = range(env.design.design, new_design, env.design_steps)
    update_design!(env, new_design)
    # propagate!(env.sim, env.sim.dt)
    propagate!(env.sim)

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