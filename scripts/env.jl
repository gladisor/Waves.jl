struct ScatteredWaveEnvState
    dim::TwoDim
    tspan::Vector{Float32}
    wave_total::AbstractArray{Float32, 3}
    wave_incident::AbstractArray{Float32, 3}
    design::AbstractDesign
end

mutable struct ScatteredWaveEnv
    wave_total::AbstractArray{Float32}
    wave_incident::AbstractArray{Float32}

    total::SplitWavePMLDynamics
    incident::SplitWavePMLDynamics{Nothing}

    σ::Vector{Float32}
    time_step::Int
    dt::Float32
    integration_steps::Int
end

Flux.@functor ScatteredWaveEnv

function Base.time(env::ScatteredWaveEnv)
    return env.time_step * env.dt
end

function (env::ScatteredWaveEnv)(action::AbstractDesign)
    ti = time(env)
    tspan = build_tspan(ti, env.dt, env.integration_steps)
    env.total = update_design(env.total, tspan, action)

    total_iter = Integrator(runge_kutta, env.total, ti, env.dt, env.integration_steps)
    u_total = unbatch(total_iter(env.wave_total))
    env.wave_total = u_total[end]

    incident_iter = Integrator(runge_kutta, env.incident, ti, env.dt, env.integration_steps)
    u_incident = unbatch(incident_iter(env.wave_incident))
    env.wave_incident = u_incident[end]

    u_scattered = u_total .- u_incident
    env.σ = sum.(energy.(displacement.(u_scattered)))

    env.time_step += env.integration_steps
end

function state(env::ScatteredWaveEnv)
    return ScatteredWaveEnvState(
        env.total.dim,
        build_tspan(time(env), env.dt, env.integration_steps),
        env.wave_total,
        env.wave_incident,
        env.total.design(time(env)))
end