
"""
Abstract type which encapsulates structures which compute rewards from WaveEnv.
"""
abstract type RewardSignal end

"""
Structure which mediates the interaction between a wave and a changing design.
The effect of the design on the wave occurs through modulation of the wave WaveSpeed
within the media.
"""
mutable struct WaveEnv <: AbstractEnv
    iter::ODEIntegrator
    C::WaveSpeed
    dt::Float64
    reward_signal::RewardSignal
end

function (rs::RewardSignal)(env::WaveEnv) end

"""
Computes the flux of a scattered wave. Contains the solution of the incident wave
"""
mutable struct ScatteredFlux <: RewardSignal
    sol_inc::ODESolution
    flux::WaveFlux
end

function (scattered_flux::ScatteredFlux)(env::WaveEnv)
    u_sc = env.iter.u[:, :, 1] .- scattered_flux.sol_inc(env.iter.t)
    return scattered_flux.flux(u_sc[:, :, 1])
end

"""
Takes the current environment and updates the WaveSpeed such that an action is applied over
a time interval. It sets the current design equal to the design at the end of the previous time interval.
"""
function update_design!(env::WaveEnv, action)
    design = DesignInterpolator(env.C.design(env.iter.t), action, env.iter.t, env.iter.t + env.dt)
    C = WaveSpeed(env.C.dim, env.C.C0, design)
    env.C = C
    env.iter.p[2] = env.C
end

"""
Propagates the wave simulation to the next time stop which is given by the environment's dt variable.
"""
function propagate_wave!(env::WaveEnv)
    add_tstop!(env.iter, env.iter.t + env.dt)
    step!(env.iter)
end

function RLBase.action_space(cyl::Cylinder)
    return Space([-1.0..1.0, -1.0..1.0])
end

function RLBase.action_space(env::WaveEnv)
    action_space(env.C.design.initial)
end

function RLBase.state(env::WaveEnv)
    return convert(Array{Float32, 3}, env.iter.u)
end

function RLBase.state_space(env::WaveEnv)
    return env.iter.u
end

function RLBase.reward(env::WaveEnv)
    return env.reward_signal(env)
end

function RLBase.is_terminated(env::WaveEnv)
    return env.iter.t >= env.iter.sol.prob.tspan[end]
end

function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    propagate_wave!(env)
end

function RLBase.reset!(env::WaveEnv)
    reinit!(env.iter)
end