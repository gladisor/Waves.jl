using CairoMakie
using Flux
using Flux.Losses: mse
using Flux: jacobian, flatten, Recur, unbatch
using Waves

function build_gradient(dim::AbstractDim)
    return Waves.gradient(dim.x)
end

function Waves.split_wave_pml(wave::AbstractMatrix{Float32}, t::Float32, C::AbstractVector{Float32}, grad::AbstractMatrix{Float32}, pml::AbstractVector{Float32})
    u = wave[:, 1] ## displacement
    v = wave[:, 2] ## velocity

    du = (C .^ 2) .* (grad * v) .- pml .* u
    dvx = grad * u .- pml .* v

    return cat(du, dvx, dims = 2)
end

function Waves.split_wave_pml(wave::AbstractArray{Float32, 3}, t::Float32, C::AbstractMatrix{Float32}, ∇::AbstractMatrix{Float32}, pml::AbstractMatrix{Float32})
    
    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    b = C .^ 2
    σx = pml
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

function Waves.split_wave_pml(
        wave::AbstractArray{Float32, 3}, 
        t::Float32, 
        design::DesignInterpolator, 
        g::AbstractArray{Float32, 3}, 
        ambient_speed::Float32, 
        ∇::AbstractMatrix{Float32}, 
        pml::AbstractMatrix{Float32})

    C = Waves.speed(design(t), g, ambient_speed)
    return split_wave_pml(wave, t, C, ∇, pml)
end

function Waves.runge_kutta(f::Function, u::AbstractArray{Float32}, t::Float32, dt::Float32, parameters...)
    k1 = f(u, t, parameters...)
    k2 = f(u .+ 0.5f0 * dt * k1, t + 0.5f0 * dt, parameters...)
    k3 = f(u .+ 0.5f0 * dt * k2, t + 0.5f0 * dt, parameters...)
    k4 = f(u .+ dt * k3,         t + dt,         parameters...)
    du = 1/6f0 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
    return u .+ du * dt
end

function build_tspan(t0::Float32, dt::Float32, steps::Int)::Vector{Float32}
    return range(t0, dt * steps, steps + 1)
end

struct Integrator
    integration_function::Function
    ode_function::Function
    dt::Float32
    parameters::Vector
end

Flux.@functor Integrator

function (iter::Integrator)(u::AbstractArray{Float32}, t::Float32, dt::Float32)
    return iter.integration_function(iter.ode_function, u, t, dt, iter.parameters...)
end

function (iter::Integrator)(u::AbstractArray{Float32}, t::Float32)
    u′ = iter(u, t, iter.dt)
    return (u′, u′)
end

function Waves.integrate(iter::Integrator, u0::AbstractArray{Float32}, t0::Float32, dt::Float32, steps::Int)
    tspan = build_tspan(t0, dt, steps-1)
    recur = Recur(iter, u0)
    u = cat(u0, [recur(t) for t in tspan]..., dims = ndims(u0) + 1)
    return u
end

elements = 256
t0 = 0.0f0
dt = 0.00001f0
steps = 1000
ambient_speed = 1531.0f0
pml_scale = ambient_speed * 10f0

dim = TwoDim(10.0f0, elements)
pulse = Pulse(dim, -4.0f0, 0.0f0, 10.0f0)
u0 = pulse(build_wave(dim, fields = 6)) |> gpu
design = DesignInterpolator(Scatterers([0.0f0 0.0f0], [1.0f0], [2120.0f0])) |> gpu
g = grid(dim)
C = ones(Float32, size(dim)...) * ambient_speed
grad = build_gradient(dim)
pml = build_pml(dim, 4.0f0, 1.0f0) .^ 2 * pml_scale

ps = [C, grad, pml]
ps = [design, g, ambient_speed, grad, pml] |> gpu

iter = Integrator(runge_kutta, split_wave_pml, dt, ps)

@time u = integrate(iter, u0, t0, dt, steps)
@time u = integrate(iter, u[:, :, :, end], t0, dt, steps)

sol = WaveSol(dim, build_tspan(t0, dt, steps), unbatch(u)) |> cpu
@time render!(sol, path = "vid.mp4", seconds = 1.0f0)

# # model = Chain(Flux.flatten, Dense(2 * elements, 1), vec)
# # y = sin.(2pi * range(0.0, 1.0, steps + 1))

# # e(x) = sum(x.^2)
# # e(u[:, 1, end])

# # gs = gradient(u0) do _u0
# #     u = integrate(iter, _u0, t0, dt, steps)
# #     e(u[:, 1, end])
# # end

# ## z = gs[:, end]
# # for i in reverse(axes(gs, 2))
# #     @time jac = jacobian(f, u[:, :, i])[1]
# #     z = z .- (z' * jac)' * dt
# # end

# # # # a = Flux.flatten(gradient(x -> mse(y, model(x)), sol)[1])
# # # # tspan = collect(range(t0, dt * steps, steps + 1))

# # # # for i in axes(sol, 3)
# # # #     @time jac = Flux.jacobian(iter, u, tspan[i], dt)[1]
# # # # end

