export WaveEnv, reset!, state, is_terminated

Base.@kwdef mutable struct WaveEnv{N <: AbstractDim, D <: AbstractDesign}
    sim::WaveSim{N}
    design::Design{D}
    design_steps::Int
end

function reset!(env::WaveEnv; kwargs...)
    reset!(env.sim)
    reset!(env.design, env.sim.wave.dim; kwargs...)
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
    ti = current_time(env.sim)
    tf = ti + env.design_steps * env.sim.dt

    set_ti!(env.sim, ti)
    set_tf!(env.sim, tf)
    add_tstop!(env.sim.iter, tf)

    new_design = perturb(env.design.design, action, env.sim.wave.dim)
    steps = range(env.design.design, new_design, env.design_steps)
    update_design!(env, new_design)
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