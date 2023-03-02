export WaveEnv

"""
Structure for handling the effect of an action on the design.

    initial_condition:  how should the wave be reset!
    sol:                the wave propagation history since the previous action
    iter:               the WaveIntegrator for propagating the wave in time
    design_space:       the set of actions to sample from
    design_steps:       number of simulation steps to apply the action over
    tmax:               the maximum physical time in the system before ending the episode.
"""
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

    u = displacement(iter.wave)
    sol = WaveSol(iter.dyn.dim, Float32[time(iter.dyn)], typeof(u)[u])
    iter.wave = initial_condition(iter.wave)

    return WaveEnv(initial_condition, sol, iter, design_space, design_steps, tmax)
end

"""
Retreive the physical time from the dynamics
"""
function Base.time(env::WaveEnv)
    return time(env.iter.dyn)
end

"""
Modify the design such that it will apply the given action over the time interval from the current
time to the time after a specified number of design_steps.
"""
function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.iter.dyn.dt
    env.iter.dyn.design = DesignInterpolator(env.iter.dyn.design(ti), action, ti, tf)
end

"""
Updates the design and procedes to propagate the wave for the specified number of design_steps
"""
function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    env.sol = integrate(env.iter, env.design_steps)
end

"""
Retreives the state of the system
"""
function RLBase.state(env::WaveEnv)
    return displacement(env.iter.wave)
end

"""
Gets all possible actions.
"""
function RLBase.action_space(env::WaveEnv)
    env.design_space
end

function RLBase.reward(env::WaveEnv)
    return 0.0f0
end

function RLBase.state_space(env::WaveEnv)
    return state(env)
end

"""
Resets the environment by reverting the wave back to the initial condition and 
resetting the time of the dynamics to zero
"""
function RLBase.reset!(env::WaveEnv)
    env.iter.wave = env.initial_condition(env.iter.wave)
    reset!(env.iter)
end

"""
Checks if the episode is over
"""
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