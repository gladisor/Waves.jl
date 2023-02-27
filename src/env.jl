export WaveEnv

mutable struct WaveEnv <: AbstractEnv
    initial_condition::InitialCondition
    sol::WaveSol
    iter::WaveIntegrator
    design_space::Union{ClosedInterval, Nothing}
    design_steps::Int
    tmax::Float32
end

function WaveEnv(;
    initial_condition::InitialCondition,
    iter::WaveIntegrator,
    design_space::Union{ClosedInterval, Nothing},
    design_steps::Int, tmax::Float32)

    sol = WaveSol(iter.dyn.dim)
    iter.wave = initial_condition(iter.wave)

    return WaveEnv(initial_condition, sol, iter, design_space, design_steps, tmax)
end

function Base.time(env::WaveEnv)
    return time(env.iter.dyn)
end

function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.iter.dyn.dt
    env.iter.dyn.design = DesignInterpolator(env.iter.dyn.design(ti), action, ti, tf)
end

function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    env.sol = integrate(env.iter, env.design_steps)
end

function RLBase.state(env::WaveEnv)
    return displacement(env.iter.wave)
end

function RLBase.action_space(env::WaveEnv)
    env.design_space
end

function RLBase.reward(env::WaveEnv)
    return 0.0f0
end

function RLBase.state_space(env::WaveEnv)
    return state(env)
end

function RLBase.reset!(env::WaveEnv)
    env.iter.wave = env.initial_condition(env.iter.wave)
    env.iter.dyn.t = 0
end

function RLBase.is_terminated(env::WaveEnv)
    return time(env) >= env.tmax
end

function Flux.gpu(env::WaveEnv)
    return WaveEnv(
        gpu(env.initial_condition),
        gpu(env.sol),
        gpu(env.iter), 
        env.design_space,
        env.design_steps, 
        env.tmax)
end

function Flux.cpu(env::WaveEnv)
    return WaveEnv(
        cpu(env.initial_condition),
        cpu(env.sol), 
        cpu(env.iter), 
        env.design_space,
        env.design_steps, 
        env.tmax)
end