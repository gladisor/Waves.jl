export WaveEnv

mutable struct WaveEnv <: AbstractEnv
    u::AbstractArray
    initial_condition::InitialCondition
    sol::WaveSol
    dyn::WaveDynamics
    design_space::Union{ClosedInterval, Nothing}
    design_steps::Int
    tmax::Float32
end

function WaveEnv(;
    initial_condition::InitialCondition,
    dyn::WaveDynamics,
    design_space::Union{ClosedInterval, Nothing},
    design_steps::Int,
    tmax::Float32)

    t = time(dyn)
    u = initial_condition(dyn.g)
    sol = WaveSol(dyn.dim, Float32[t], typeof(u)[u])

    return WaveEnv(u, initial_condition, sol, dyn, design_space, design_steps, tmax)
end

function Base.time(env::WaveEnv)
    return time(env.dyn)
end

function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.dyn.dt
    env.dyn.design = DesignInterpolator(env.dyn.design(ti), action, ti, tf)
end

function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    env.sol = integrate(env.u, env.dyn, env.design_steps)
    env.u = env.sol.u[end]
end

function RLBase.state(env::WaveEnv)
    return env.u
end

function RLBase.action_space(env::WaveEnv)
    env.design_space
end

function RLBase.reward(env::WaveEnv)
    return 0.0f0
end

function RLBase.state_space(env::WaveEnv)
    return env.u
end

function RLBase.reset!(env::WaveEnv)
    env.u = env.initial_condition(env.dyn.g)
    env.dyn.t = 0
end

function RLBase.is_terminated(env::WaveEnv)
    return time(env) >= env.tmax
end

function Flux.gpu(env::WaveEnv)
    return WaveEnv(
        gpu(env.u), 
        gpu(env.initial_condition),
        gpu(env.sol),
        gpu(env.dyn), 
        env.design_space,
        env.design_steps, 
        env.tmax)
end

function Flux.cpu(env::WaveEnv)
    return WaveEnv(
        cpu(env.u), 
        cpu(env.initial_condition),
        cpu(env.sol), 
        cpu(env.dyn), 
        env.design_space,
        env.design_steps, 
        env.tmax)
end