export WaveEnvState
export WaveImage, DisplacementImage
export WaveEnv, RandomDesignPolicy

struct WaveEnvState
    dim::TwoDim
    tspan::Vector{Float32}
    wave_total::AbstractArray{Float32, 3}
    wave_incident::AbstractArray{Float32, 3}
    design::AbstractDesign
end

Flux.@functor WaveEnvState

struct WaveImage <: AbstractSensor end
(sensor::WaveImage)(s::WaveEnvState) = s

struct DisplacementImage <: AbstractSensor end

function (sensor::DisplacementImage)(s::WaveEnvState)
    return WaveEnvState(
        s.dim, s.tspan,
        s.wave_total[:, :, 1, :], s.wave_incident[:, :, 1, :],
        s.design)
end

mutable struct WaveEnv <: AbstractEnv
    dim::AbstractDim
    reset_wave::AbstractInitialWave

    design_space::DesignSpace
    action_speed::Float32 ## speed that action is applied over m/s

    sensor::AbstractSensor

    wave_total::AbstractArray{Float32}
    wave_incident::AbstractArray{Float32}

    total_dynamics::WaveDynamics
    incident_dynamics::WaveDynamics

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
        design_space::DesignSpace,
        action_speed::Float32 = 500.0f0,
        source::AbstractSource = NoSource(),
        sensor::AbstractSensor = WaveImage(),
        ambient_speed::Float32 = AIR,
        pml_width::Float32 = 2.0f0,
        pml_scale::Float32 = 20000.0f0,
        dt::Float32 = Float32(1e-5),
        integration_steps::Int = 100,
        actions::Int = 10
        )

    wave = reset_wave(build_wave(dim, fields = 6))
    total_dynamics = WaveDynamics(dim, ambient_speed = ambient_speed, pml_width = pml_width, pml_scale = pml_scale, design = rand(design_space), source = source)
    incident_dynamics = WaveDynamics(dim, ambient_speed = ambient_speed, pml_width = pml_width, pml_scale = pml_scale, design = NoDesign(), source = source)
    sigma = zeros(Float32, integration_steps + 1)
    return WaveEnv(dim, reset_wave, design_space, action_speed, sensor, wave, wave, total_dynamics, incident_dynamics, sigma, 0, dt, integration_steps, actions)
end

function Base.time(env::WaveEnv)
    return env.time_step * env.dt
end

function build_tspan(env::WaveEnv) 
    return build_tspan(time(env), env.dt, env.integration_steps)
end

function RLBase.is_terminated(env::WaveEnv)
    return env.time_step >= env.actions * env.integration_steps
end

function RLBase.reset!(env::WaveEnv)
    env.time_step = 0
    env.wave_total = gpu(env.reset_wave(env.wave_total))
    env.wave_incident = gpu(env.reset_wave(env.wave_incident))
    design = gpu(DesignInterpolator(rand(env.design_space)))

    env.total_dynamics = WaveDynamics(
        env.total_dynamics.ambient_speed, 
        design, 
        env.total_dynamics.source, 
        env.total_dynamics.grid, 
        env.total_dynamics.grad, 
        env.total_dynamics.pml, 
        env.total_dynamics.bc)

    env.σ = zeros(Float32, env.integration_steps + 1)
    return nothing
end

function (env::WaveEnv)(action::AbstractDesign)
    ti = time(env)
    tspan = build_tspan(ti, env.dt, env.integration_steps)
    env.total_dynamics = update_design(env.total_dynamics, tspan, env.design_space, gpu(action))

    total_iter = Integrator(runge_kutta, env.total_dynamics, ti, env.dt, env.integration_steps)
    u_total = unbatch(total_iter(env.wave_total))
    env.wave_total = u_total[end]

    incident_iter = Integrator(runge_kutta, env.incident_dynamics, ti, env.dt, env.integration_steps)
    u_incident = unbatch(incident_iter(env.wave_incident))
    env.wave_incident = u_incident[end]

    u_scattered = u_total .- u_incident
    
    env.σ = sum.(energy.(displacement.(u_scattered))) / 64.0f0

    env.time_step += env.integration_steps
    return (tspan, u_incident, u_scattered)
end

function RLBase.state(env::WaveEnv)
    return env.sensor(WaveEnvState(
        env.dim,
        build_tspan(env),
        env.wave_total,
        env.wave_incident,
        env.total_dynamics.design(time(env))))
end

function RLBase.state_space(env::WaveEnv)
    return state(env)
end

function RLBase.action_space(env::WaveEnv)
    return build_action_space(rand(env.design_space), env.action_speed * env.dt * env.integration_steps)
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
    # action::ClosedInterval{<: AbstractDesign}
    a_space::DesignSpace
end

function (policy::RandomDesignPolicy)(::WaveEnv)
    return rand(policy.a_space)
end