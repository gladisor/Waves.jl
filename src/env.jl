
export WaveEnv

mutable struct WaveEnv
    u::AbstractArray
    dyn::WaveDynamics
    design_steps::Int
end

function Base.time(env::WaveEnv)
    return env.dyn.t * env.dyn.dt
end

function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.dyn.dt
    env.dyn.C.design = DesignInterpolator(env.dyn.C.design(ti), action, ti, tf)
    return nothing
end

function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    sol = integrate(env.u, env.dyn, env.design_steps)
    env.u = sol.u[end]
    return sol
end

function Flux.gpu(env::WaveEnv)
    u = gpu(env.u)
    dyn = gpu(env.dyn)
    return WaveEnv(u, dyn, env.design_steps)
end