struct ScatteredWaveEnvState
    dim::TwoDim
    tspan::Vector{Float32}
    wave_total::AbstractArray{Float32, 3}
    wave_incident::AbstractArray{Float32, 3}
    design::AbstractDesign
end

Flux.@functor ScatteredWaveEnvState

mutable struct ScatteredWaveEnv <: AbstractEnv
    initial_condition::AbstractInitialCondition

    wave_total::AbstractArray{Float32}
    wave_incident::AbstractArray{Float32}

    total_dynamics::SplitWavePMLDynamics
    incident_dynamics::SplitWavePMLDynamics{Nothing}

    reset_design::Function
    action_space::ClosedInterval

    σ::Vector{Float32}
    time_step::Int
    dt::Float32
    integration_steps::Int
    actions::Int
end

Flux.@functor ScatteredWaveEnv

function ScatteredWaveEnv(
        dim::TwoDim;
        initial_condition::AbstractInitialCondition,
        design::AbstractDesign,
        ambient_speed::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        reset_design::Function,
        action_space::ClosedInterval,
        dt::Float32 = Float32(5e-5),
        integration_steps::Int = 100,
        actions::Int = 10
        )

    design = DesignInterpolator(design)

    grid = build_grid(dim)
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    wave = initial_condition(build_wave(dim, fields = 6))

    total_dynamics = SplitWavePMLDynamics(design, dim, grid, ambient_speed, grad, bc, pml)
    incident_dynamics = SplitWavePMLDynamics(nothing, dim, grid, ambient_speed, grad, bc, pml)

    sigma = zeros(Float32, steps + 1)

    return ScatteredWaveEnv(
        initial_condition, 
        wave, wave,
        total_dynamics, incident_dynamics,
        reset_design, action_space, sigma,
        0, dt, integration_steps, actions
        )
end

function Base.time(env::ScatteredWaveEnv)
    return env.time_step * env.dt
end

function RLBase.is_terminated(env::ScatteredWaveEnv)
    return env.time_step >= env.actions * env.integration_steps
end

function RLBase.reset!(env::ScatteredWaveEnv)
    env.time_step = 0
    env.wave_total = gpu(env.initial_condition(env.wave_total))
    env.wave_incident = gpu(env.initial_condition(env.wave_incident))

    initial = env.reset_design(env.total_dynamics.design.initial)

    env.total_dynamics = SplitWavePMLDynamics(
        DesignInterpolator(initial),
        env.total_dynamics.dim,
        env.total_dynamics.grid,
        env.total_dynamics.ambient_speed,
        env.total_dynamics.grad,
        env.total_dynamics.bc,
        env.total_dynamics.pml)

    env.σ = zeros(Float32, env.integration_steps + 1)
    return nothing
end

function (env::ScatteredWaveEnv)(action::AbstractDesign)
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
end

function RLBase.state(env::ScatteredWaveEnv)
    return ScatteredWaveEnvState(
        env.total_dynamics.dim,
        build_tspan(time(env), env.dt, env.integration_steps),
        env.wave_total,
        env.wave_incident,
        env.total_dynamics.design(time(env)))
end

function RLBase.state_space(env::ScatteredWaveEnv)
    return state(env)
end

function RLBase.action_space(env::ScatteredWaveEnv)
    return env.action_space
end

function RLBase.reward(env::ScatteredWaveEnv)
    return sum(env.σ)
end

function episode_trajectory(env::ScatteredWaveEnv)
    traj = CircularArraySARTTrajectory(
        capacity = env.max_steps,
        state = Vector{ScatteredWaveEnv} => (),
        action = Vector{typeof(env.total_dynamics.design.initial)} => ())

    return traj
end

mutable struct RandomDesignPolicy <: AbstractPolicy
    action::ClosedInterval{<: AbstractDesign}
end

function (policy::RandomDesignPolicy)(::ScatteredWaveEnv)
    return rand(policy.action)
end