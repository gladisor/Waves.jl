struct WaveEnvState
    dim::TwoDim
    tspan::Vector{Float32}
    wave_total::AbstractArray{Float32, 3}
    wave_incident::AbstractArray{Float32, 3}
    design::AbstractDesign
end

abstract type AbstractSensor end

struct WaveImage <: AbstractSensor end
(sensor::WaveImage)(s::WaveEnvState) = s

struct DisplacementImage <: AbstractSensor end

function (sensor::DisplacementImage)(s::WaveEnvState)
    return WaveEnvState(
        s.dim, s.tspan,
        s.wave_total[:, :, 1, :], s.wave_incident[:, :, 1, :],
        s.design)
end

Flux.@functor WaveEnvState

mutable struct WaveEnv <: AbstractEnv
    reset_wave::AbstractInitialWave
    reset_design::AbstractInitialDesign
    action_space::ClosedInterval

    sensor::AbstractSensor

    wave_total::AbstractArray{Float32}
    wave_incident::AbstractArray{Float32}

    total_dynamics::SplitWavePMLDynamics
    incident_dynamics::SplitWavePMLDynamics

    σ::Vector{Float32}
    time_step::Int
    dt::Float32
    integration_steps::Int
    actions::Int
end

Flux.@functor WaveEnv

function WaveEnv(
        dim::TwoDim;
        reset_wave::AbstractInitialWave,
        reset_design::AbstractInitialDesign,
        action_space::ClosedInterval,
        source::AbstractSource = NoSource(),
        sensor::AbstractSensor = WaveImage(),
        ambient_speed::Float32 = AIR,
        pml_width::Float32 = 2.0f0,
        pml_scale::Float32 = 20000.0f0,
        dt::Float32 = Float32(5e-5),
        integration_steps::Int = 100,
        actions::Int = 10
        )

    # design = DesignInterpolator(reset_design())

    # grid = build_grid(dim)
    # grad = build_gradient(dim)
    # bc = dirichlet(dim)
    # pml = build_pml(dim, pml_width, pml_scale)
    wave = reset_wave(build_wave(dim, fields = 6))

    # total_dynamics = SplitWavePMLDynamics(
    #     design, dim, grid, ambient_speed, grad, bc, pml)
        
    # incident_dynamics = SplitWavePMLDynamics(
    #     DesignInterpolator(NoDesign()), 
    #     dim, grid, ambient_speed, grad, bc, pml)

    total_dynamics = WaveDynamics(
        dim,
        ambient_speed = ambient_speed,
        pml_width = pml_width, pml_scale = pml_scale, 
        design = reset_design(), source = source)

    incident_dynamics = WaveDynamics(
        dim,
        ambient_speed = ambient_speed,
        pml_width = pml_width, pml_scale = pml_scale,
        design = NoDesign(), source = source)

    sigma = zeros(Float32, steps + 1)

    return WaveEnv(
        reset_wave, reset_design, action_space, sensor,
        wave, wave, total_dynamics, incident_dynamics, 
        sigma, 0, dt, integration_steps, actions)
end

function Base.time(env::WaveEnv)
    return env.time_step * env.dt
end

function RLBase.is_terminated(env::WaveEnv)
    return env.time_step >= env.actions * env.integration_steps
end

function RLBase.reset!(env::WaveEnv)
    env.time_step = 0
    env.wave_total = gpu(env.reset_wave(env.wave_total))
    env.wave_incident = gpu(env.reset_wave(env.wave_incident))
    design = gpu(DesignInterpolator(env.reset_design()))
    env.total_dynamics = SplitWavePMLDynamics(
        design,
        env.total_dynamics.dim,
        env.total_dynamics.grid,
        env.total_dynamics.ambient_speed,
        env.total_dynamics.grad,
        env.total_dynamics.bc,
        env.total_dynamics.pml)

    env.σ = zeros(Float32, env.integration_steps + 1)
    return nothing
end

function (env::WaveEnv)(action::AbstractDesign)
    ti = time(env)
    tspan = build_tspan(ti, env.dt, env.integration_steps)
    env.total_dynamics = update_design(env.total_dynamics, tspan, gpu(action))

    total_iter = Integrator(runge_kutta, env.total_dynamics, ti, env.dt, env.integration_steps)
    u_total = unbatch(total_iter(env.wave_total))
    env.wave_total = u_total[end]

    incident_iter = Integrator(runge_kutta, env.incident_dynamics, ti, env.dt, env.integration_steps)
    u_incident = unbatch(incident_iter(env.wave_incident))
    env.wave_incident = u_incident[end]

    u_scattered = u_total .- u_incident
    env.σ = sum.(energy.(displacement.(u_scattered)))

    env.time_step += env.integration_steps

    return tspan, u_total
end

function RLBase.state(env::WaveEnv)
    return env.sensor(WaveEnvState(
        env.total_dynamics.dim,
        build_tspan(time(env), env.dt, env.integration_steps),
        env.wave_total,
        env.wave_incident,
        env.total_dynamics.design(time(env))))
end

function RLBase.state_space(env::WaveEnv)
    return state(env)
end

function RLBase.action_space(env::WaveEnv)
    return env.action_space
end

function RLBase.reward(env::WaveEnv)
    return sum(env.σ)
end

function episode_trajectory(env::WaveEnv)
    traj = CircularArraySARTTrajectory(
        capacity = env.actions,
        state = Vector{WaveEnvState} => (),
        action = Vector{typeof(env.reset_design())} => ())

    return traj
end

mutable struct RandomDesignPolicy <: AbstractPolicy
    action::ClosedInterval{<: AbstractDesign}
end

function (policy::RandomDesignPolicy)(::WaveEnv)
    return rand(policy.action)
end