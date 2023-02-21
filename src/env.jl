export WaveEnv

mutable struct WaveEnv <: AbstractEnv
    u::AbstractArray
    initial_condition::InitialCondition
    sol::WaveSol
    dyn::WaveDynamics
    design_steps::Int
    tmax::Float32
end

function WaveEnv(;
    initial_condition::InitialCondition,
    dyn::WaveDynamics,
    design_steps::Int,
    tmax::Float32)

    sol = WaveSol(dyn.dim, Float32[], AbstractArray{Float32}[])

    return WaveEnv(
        initial_condition(dyn.dim), 
        initial_condition, 
        sol, dyn, design_steps, tmax)
end

function Base.time(env::WaveEnv)
    return env.dyn.t * env.dyn.dt
end

function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.dyn.dt
    env.dyn.C.design = DesignInterpolator(env.dyn.C.design(ti), action, ti, tf)
end

function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    env.sol = integrate(env.u, env.dyn, env.design_steps)
    env.u = env.sol.u[end]
end

function RLBase.state(env::WaveEnv)
    return env.u
end

function RLBase.action_space(env::WaveEnv, args...)
    return action_space(env.dyn.C.design.initial, args...)
end

function RLBase.reward(env::WaveEnv)
    return 0.0f0
end

function RLBase.state_space(env::WaveEnv)
    return env.u
end

function RLBase.reset!(env::WaveEnv)
    env.u = env.initial_condition(env.dyn.dim)
    env.dyn.t = 0
end

function RLBase.is_terminated(env::WaveEnv)
    return time(env) >= env.tmax
end

function Flux.gpu(env::WaveEnv)
    return WaveEnv(
        gpu(env.u), 
        env.initial_condition,
        env.sol, # gpu(env.sol),
        gpu(env.dyn), 
        env.design_steps, 
        env.tmax)
end

function Flux.cpu(env::WaveEnv)
    return WaveEnv(
        cpu(env.u), 
        env.initial_condition,
        cpu(env.sol), 
        cpu(env.dyn), 
        env.design_steps, 
        env.tmax)
end