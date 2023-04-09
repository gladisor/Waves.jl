
mutable struct ScatteredWaveEnv
    wave_total::AbstractArray{Float32}
    wave_incident::AbstractArray{Float32}

    total::SplitWavePMLDynamics
    incident::SplitWavePMLDynamics

    σ::Vector{Float32}
    time_step::Int
    dt::Float32
    integration_steps::Int
end

Flux.@functor ScatteredWaveEnv

function Base.time(env::WaveEnv)
    return env.time_step * env.dt
end

function (env::ScatteredWaveEnv)(action::AbstractDesign)
    tspan = build_tspan(time(env), env.dt, env.integration_steps)
    env.total = update_design(env.total, tspan, action)

    total_iter = Integrator(runge_kutta, env.total, env.dt)
    u_total = unbatch(integrate(total_iter, env.wave_total, time(env), env.integration_steps))
    env.wave_total = u_total[end]

    incident_iter = Integrator(runge_kutta, env.incident, env.dt)
    u_incident = unbatch(integrate(incident_iter, env.wave_incident, time(env), env.integration_steps))
    env.wave_incident = u_incident[end]

    u_scattered = u_total .- u_incident
    env.σ = sum.(energy.(displacement.(u_scattered)))

    env.time_step += env.integration_steps
end

function CairoMakie.plot(env::ScatteredWaveEnv)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)
    heatmap!(ax, cpu(env.total.dim.x), cpu(env.total.dim.y), cpu(displacement(env.wave_total)), colormap = :ice)
    mesh!(ax, cpu(env.total.design(time(env))))
    return fig
end


#=
Example of using ScatteredWaveEnv to measure and plot the scattered wave caused
by some actions.
=#

dim = TwoDim(grid_size, elements)
g = grid(dim)
C = ones(Float32, size(dim)...) * ambient_speed
grad = build_gradient(dim)
pml = build_pml(dim, pml_width, pml_scale)

pulse = Pulse(dim, -5.0f0, 0.0f0, pulse_intensity)
wave = pulse(build_wave(dim, fields = 6))

initial = Scatterers([0.0f0 0.0f0], [1.0f0], [3100.0f0])
policy = radii_design_space(initial, 1.0f0)
design = DesignInterpolator(initial)

env = ScatteredWaveEnv(
    wave, wave,
    SplitWavePMLDynamics(design, dim, g, ambient_speed, grad, pml), 
    SplitWavePMLDynamics(nothing, dim, g, ambient_speed, grad, pml),
    zeros(Float32, steps),
    0, dt, steps) |> gpu

e = []

iterations = 20
for i in 1:iterations

    @time env(gpu(rand(policy)))

    fig = plot(env)
    save("u_$i.png", fig)

    push!(e, env.σ)
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, cpu(vcat(e...)))
save("energy.png", fig)