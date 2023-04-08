using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, unbatch, pullback
using Waves

function build_gradient(dim::AbstractDim)
    return Waves.gradient(dim.x)
end

function Waves.runge_kutta(f::AbstractDynamics, u::AbstractArray{Float32}, t::Float32, dt::Float32)
    k1 = f(u, t)
    k2 = f(u .+ 0.5f0 * dt * k1, t + 0.5f0 * dt)
    k3 = f(u .+ 0.5f0 * dt * k2, t + 0.5f0 * dt)
    k4 = f(u .+ dt * k3,         t + dt)
    du = 1/6f0 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
    return u .+ du * dt
end

function build_tspan(ti::Float32, dt::Float32, steps::Int)::Vector{Float32}
    return range(ti, ti + steps*dt, steps + 1)
end

struct Integrator
    integration_function::Function
    dynamics::AbstractDynamics
    dt::Float32
end

Flux.@functor Integrator

function (iter::Integrator)(u::AbstractArray{Float32}, t::Float32, dt::Float32)
    return iter.integration_function(iter.dynamics, u, t, dt)
end

function (iter::Integrator)(u::AbstractArray{Float32}, t::Float32)
    u′ = iter(u, t, iter.dt)
    return (u′, u′)
end

function Waves.integrate(iter::Integrator, ui::AbstractArray{Float32}, ti::Float32, steps::Int)
    tspan = build_tspan(ti, iter.dt, steps-1)
    recur = Recur(iter, ui)
    u = cat(ui, [recur(t) for t in tspan]..., dims = ndims(ui) + 1)
    return u
end

function Waves.speed(::Nothing, g::AbstractArray{Float32}, ambient_speed::Float32)
    return dropdims(sum(g, dims = ndims(g)), dims = ndims(g)) .^ 0
end

struct SplitWavePMLDynamics{D <: Union{DesignInterpolator, Nothing}} <: AbstractDynamics
    design::D
    dim::AbstractDim
    g::AbstractArray{Float32}
    ambient_speed::Float32
    grad::AbstractArray{Float32}
    pml::AbstractArray{Float32}
end

Flux.@functor SplitWavePMLDynamics

function Waves.speed(dynamics::SplitWavePMLDynamics{Nothing}, t::Float32)
    return dynamics.ambient_speed
end

function Waves.speed(dynamics::SplitWavePMLDynamics{DesignInterpolator}, t::Float32)
    return speed(dynamics.design(t), dynamics.g, dynamics.ambient_speed)
end

function update_design(dynamics::SplitWavePMLDynamics{DesignInterpolator}, tspan::Vector{Float32}, action::AbstractDesign)::SplitWavePMLDynamics{DesignInterpolator}
    initial = dynamics.design(tspan[1])
    design = DesignInterpolator(initial, action, tspan[1], tspan[end])
    return SplitWavePMLDynamics(design, dynamics.dim, dynamics.g, dynamics.ambient_speed, dynamics.grad, dynamics.pml)
end

function (dyn::SplitWavePMLDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    C = dyn.ambient_speed
    ∇ = dyn.grad
    σ = dyn.pml

    du = C ^ 2 * (∇ * v) .- σ .* u
    dv = ∇ * u .- σ .* v

    return hcat(du, dv)
end

function (dyn::SplitWavePMLDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)

    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    C = speed(dyn, t)
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'

    Vxx = ∇ * Vx
    Vyy = (∇ * Vy')'
    Ux = ∇ * U
    Uy = (∇ * U')'

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

struct LinearWave <: AbstractDynamics
    C::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor LinearWave

function (dyn::LinearWave)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    du =  (dyn.C .^ 2) .* (dyn.grad * v) .* dyn.bc
    dv = dyn.grad * u
    return hcat(du, dv)
end

mutable struct WaveEnv
    wave_total::AbstractArray{Float32}
    wave_incident::AbstractArray{Float32}

    total::SplitWavePMLDynamics
    incident::SplitWavePMLDynamics

    σ::Vector{Float32}
    time_step::Int
    dt::Float32
    integration_steps::Int
end

Flux.@functor WaveEnv

function Base.time(env::WaveEnv)
    return env.time_step * env.dt
end

function (env::WaveEnv)(action::AbstractDesign)
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

function CairoMakie.plot(env::WaveEnv)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)
    heatmap!(ax, cpu(env.total.dim.x), cpu(env.total.dim.y), cpu(displacement(env.wave_total)), colormap = :ice)
    mesh!(ax, cpu(env.total.design(time(env))))
    return fig
end

grid_size = 10.f0
elements = 512
dt = 0.00002f0
steps = 50
ambient_speed = 1531.0f0
pulse_intensity = 1.0f0
pml_width = 2.0f0
pml_scale = ambient_speed * 50.0f0

dim = TwoDim(grid_size, elements)
g = grid(dim)
C = ones(Float32, size(dim)...) * ambient_speed
grad = build_gradient(dim)
pml = build_pml(dim, pml_width, pml_scale)

pulse = Pulse(dim, -5.0f0, 0.0f0, pulse_intensity)
wave = pulse(build_wave(dim, fields = 6))

# initial = Scatterers([2.0f0 0.0f0], [1.0f0], [2120.0f0])
initial = Scatterers([2.0f0 0.0f0], [1.0f0], [3100.0f0])
# action = Scatterers([-0.1f0 0.1f0], [0.00f0], [0.0f0])
policy = design_space(initial, 1.0f0)
design = DesignInterpolator(initial)

env = WaveEnv(
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

# tspan = build_tspan(time(env), dt, steps)
# env.total = update_design(env.total, tspan, action)
# iter = Integrator(runge_kutta, env.total, env.dt)
# u = integrate(iter, env.wave, time(env), env.integration_steps)
# v, back = pullback(_wave -> iter(_wave, ti)[1], wave)

# # total.design(tspan[1])
# bc = ones(Float32, size(C))
# bc[[1, end]] .= 0.0f0
# dynamics = gpu(LinearWave(C, grad, bc))

# iter = Integrator(runge_kutta, dynamics, dt)
# @time u = integrate(iter, wave, ti, steps);

# l, back = pullback(_u -> sum(_u[:, 1, :] .^ 2), u)
# dloss = back(1.0f0)[1]

# total_adj = [dloss[:, :, end]]

# tspan = build_tspan(ti, dt, steps)

# for i in reverse(axes(u, 3))
#     _, back = pullback(_wave -> runge_kutta(dynamics, _wave, ti, dt), u[:, :, i])
#     adj = - back(dloss[:, :, i])[1]
#     push!(total_adj, adj)
# end

# total_adj = sum(total_adj) * dt

# using CairoMakie

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, total_adj[:, 2])
# save("grad.png", fig)

# # # sol = WaveSol(dim, build_tspan(ti, dt, steps), unbatch(u)) |> cpu
# # # ;
# # # # DesignTrajectory(design, steps)
# # # @time render!(sol, path = "vid.mp4", seconds = 1.0f0)
