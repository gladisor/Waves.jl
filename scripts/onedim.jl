using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, unbatch
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

function build_tspan(t0::Float32, dt::Float32, steps::Int)::Vector{Float32}
    return range(t0, t0 + steps*dt, steps + 1)
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

function Waves.integrate(iter::Integrator, u0::AbstractArray{Float32}, t0::Float32, steps::Int)
    tspan = build_tspan(t0, iter.dt, steps-1)
    recur = Recur(iter, u0)
    u = cat(u0, [recur(t) for t in tspan]..., dims = ndims(u0) + 1)
    return u
end

function Waves.speed(::Nothing, g::AbstractArray{Float32}, ambient_speed::Float32)
    return dropdims(sum(g, dims = ndims(g)), dims = ndims(g)) .^ 0
end

struct SplitWavePMLDynamics <: AbstractDynamics
    design::DesignInterpolator
    g::AbstractArray{Float32}
    ambient_speed::Float32
    grad::AbstractArray{Float32}
    pml::AbstractArray{Float32}
end

Flux.@functor SplitWavePMLDynamics

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

    C = speed(dyn.design(t), dyn.g, dyn.ambient_speed)
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

struct NonLinearWave <: AbstractDynamics
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
    mlp::Chain
end

grid_size = 10.f0
elements = 512
t0 = 0.0f0
dt = 0.00002f0
steps = 500
tf = steps * dt
ambient_speed = 1531.0f0
pml_width = 2.0f0
pml_scale = ambient_speed * 50f0

dim = OneDim(grid_size, elements)
pulse = Pulse(dim, -4.0f0, 1.0f0)
u0 = pulse(build_wave(dim, fields = 2)) |> gpu

design = Scatterers([2.0f0 0.0f0], [2.0f0], [2120.0f0])
action = Scatterers([0.0f0 0.0f0], [1.0f0], [0.0f0])
design = DesignInterpolator(design, action, t0, tf) |> gpu

g = grid(dim)
C = ones(Float32, size(dim)...) * ambient_speed
grad = build_gradient(dim)
pml = build_pml(dim, pml_width, pml_scale)

# dynamics = gpu(SplitWavePMLDynamics(design, g, ambient_speed, grad, pml))
bc = ones(Float32, size(C))
bc[[1, end]] .= 0.0f0
dynamics = gpu(LinearWave(C, grad, bc))

iter = Integrator(runge_kutta, dynamics, dt)
@time u = integrate(iter, u0, t0, steps)
@time u = integrate(iter, u[:, :, end], tf, steps)
@time u = integrate(iter, u[:, :, end], tf, steps)

sol = WaveSol(dim, build_tspan(t0, dt, steps), unbatch(u)) |> cpu
# DesignTrajectory(design, steps)
@time render!(sol, path = "vid.mp4", seconds = 5.0f0)
