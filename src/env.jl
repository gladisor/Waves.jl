export WaveEnv, WaveEnvState, num_steps

"""
Structure for handling the effect of an action on the design.

    initial_condition:  how should the wave be reset!
    sol:                the wave propagation history since the previous action
    design_space:       the set of actions to sample from
    design_steps:       number of simulation steps to apply the action over
    tmax:               the maximum physical time in the system before ending the episode.
"""
mutable struct WaveEnv <: AbstractEnv
    initial_condition::InitialCondition
    sol::TotalWaveSol

    cell::AbstractWaveCell
    total_dynamics::WaveDynamics
    incident_dynamics::WaveDynamics

    space::Union{ClosedInterval, Nothing}
    design_steps::Int
    tmax::Float32
end

function WaveEnv(;
        initial_condition::InitialCondition, 
        wave::AbstractArray{Float32}, 
        cell::AbstractWaveCell,
        design::AbstractDesign,
        space::Union{ClosedInterval, Nothing},
        design_steps::Int,
        tmax::Float32,
        dynamics_kwargs...)

    total_dynamics = WaveDynamics(design = design; dynamics_kwargs...)
    incident_dynamics = WaveDynamics(;dynamics_kwargs...)

    Waves.reset!(initial_condition)

    wave = initial_condition(wave)
    total = solve(cell, wave, total_dynamics, 0)
    incident = solve(cell, wave, incident_dynamics, 0)
    sol = TotalWaveSol(total = total, incident = incident)

    return WaveEnv(initial_condition, sol, cell, total_dynamics, incident_dynamics, space, design_steps, tmax)
end

"""
Retreive the physical time from the dynamics
"""
function Base.time(env::WaveEnv)
    return time(env.total_dynamics)
end

"""
Modify the design such that it will apply the given action over the time interval from the current
time to the time after a specified number of design_steps.
"""
function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.total_dynamics.dt
    env.total_dynamics.design = DesignInterpolator(env.total_dynamics.design(ti), action, ti, tf)
end

"""
Updates the design and procedes to propagate the wave for the specified number of design_steps
"""
function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    total = solve(env.cell, env.sol.total.u[end], env.total_dynamics, env.design_steps)
    incident = solve(env.cell, env.sol.incident.u[end], env.incident_dynamics, env.design_steps)
    env.sol = TotalWaveSol(total = total, incident = incident)
end


struct WaveEnvState{D <: AbstractDesign}
    sol::TotalWaveSol
    design::D
end

function Flux.gpu(s::WaveEnvState)
    return WaveEnvState(gpu(s.sol), gpu(s.design))
end

function Flux.cpu(s::WaveEnvState)
    return WaveEnvState(cpu(s.sol), cpu(s.design))
end

"""
Retreives the state of the system
"""
function RLBase.state(env::WaveEnv)
    return cpu(WaveEnvState(env.sol, initial_design(env.total_dynamics.design)))
end

"""
Gets all possible actions.
"""
function RLBase.action_space(env::WaveEnv)
    env.space
end

function RLBase.reward(env::WaveEnv)
    return -sum(sum.(energy.(displacement.(env.sol.scattered.u))))
end

function RLBase.state_space(env::WaveEnv)
    return state(env)
end

"""
Resets the environment by reverting the wave back to the initial condition and 
resetting the time of the dynamics to zero
"""
function RLBase.reset!(env::WaveEnv)
    Waves.reset!(env.initial_condition)
    Waves.reset!(env.total_dynamics)
    Waves.reset!(env.incident_dynamics)

    wave = env.initial_condition(env.sol.total.u[end])

    total = solve(env.cell, wave, env.total_dynamics, 0)
    incident = solve(env.cell, wave, env.incident_dynamics, 0)
    env.sol = TotalWaveSol(total = total, incident = incident)
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
        gpu(env.cell),
        gpu(env.total_dynamics), 
        gpu(env.incident_dynamics),

        env.space,
        env.design_steps, 
        env.tmax)
end

function Flux.cpu(env::WaveEnv)
    return WaveEnv(
        cpu(env.initial_condition),
        cpu(env.sol), 
        cpu(env.cell),
        cpu(env.total_dynamics), 
        cpu(env.incident_dynamics),

        env.space,
        env.design_steps, 
        env.tmax)
end

function num_steps(env::WaveEnv)
    return Int(round(env.tmax / env.total_dynamics.dt))
end

function DesignTrajectory(env::WaveEnv)
    return DesignTrajectory(env.total_dynamics, env.design_steps)
end