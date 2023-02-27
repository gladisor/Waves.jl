using Waves
using Waves: split_wave_pml_2d, runge_kutta
using CairoMakie

function build_pml_x(dim::TwoDim, width::Float32, scale::Float32)
    x = abs.(dim.x)
    pml_start = x[1] - width
    pml_region = x .> pml_start
    x[.~ pml_region] .= 0.0f0
    x[pml_region] .= (x[pml_region] .- minimum(x[pml_region])) / (maximum(x[pml_region]) - minimum(x[pml_region]))
    x = repeat(x, 1, length(dim.y))
    return x .^ 2 * scale
end

"""
Comprised of two functions:
    f(u, dyn) = du/dt
    integration_function(f, u, dyn)
"""
mutable struct WaveIntegrator
    f::Function
    integration_function::Function
    dyn::WaveDynamics
end

function Base.step(iter::WaveIntegrator, u)
    u = iter.integration_function(iter.f, u, iter.dyn)
    iter.dyn.t += 1
    return u
end

function Waves.integrate(iter::WaveIntegrator, u, n::Int)
    t = Float32[time(iter.dyn)]
    sol = typeof(u)[u]
    tf = iter.dyn.t + n

    while iter.dyn.t < tf
        u = step(iter, u)
        push!(t, time(iter.dyn))
        push!(sol, u)
    end

    return WaveSol(iter.dyn.dim, t, sol)
end

function wave(u::AbstractMatrix{Float32}, t::Float32, dyn::WaveDynamics)
    U = view(u, :, 1)
    Vx = view(u, :, 2)

    U[[1, end]] .= 0.0f0

    C = dyn.ambient_speed

    σx = dyn.pml
    ∇ = dyn.grad
    dU = C^2 * ∇ * Vx .- σx .* U
    dVx = ∇ * U .- σx .* Vx

    return hcat(dU, dVx)
end

function wave(u::AbstractArray{Float32, 3}, t::Float32, dyn::WaveDynamics)
    U = view(u, :, :, 1)
    Vx = view(u, :, :, 2)
    Vy = view(u, :, :, 3)
    Ψx = view(u, :, :, 4)
    Ψy = view(u, :, :, 5)
    Ω = view(u, :, :, 6)

    U[:, [1, end]] .= 0.0f0
    U[[1, end], :] .= 0.0f0

    C = dyn.ambient_speed
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'

    Vxx = ∇ * Vx
    Vyy = (∇ * Vy')'
    Ux = ∇ * U
    Uy = (∇ * U')'

    dU = C^2 * (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = C^2 * Ux .- σx .* Vx
    dVy = C^2 * Uy .- σy .* Vy
    dΨx = C^2 * σx .* Vyy
    dΨy = C^2 * σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

dim = TwoDim(5.0f0, 200)

pml_width = 1.0f0
pml_scale = 50.0f0
ambient_speed = 1.0f0
dt = 0.01f0

dyn = WaveDynamics(dim = dim, pml_width = pml_width, pml_scale = pml_scale, ambient_speed = ambient_speed, dt = dt)
dyn.pml = build_pml_x(dim, pml_width, pml_scale)
u = exp.(-5.0f0 * dropdims(sum((dyn.g .- [0.0f0;;;0.0f0]).^ 2, dims = 3), dims = 3))

u0 = cat(u, zeros(Float32, size(u)..., 5), dims = 3)
iter = WaveIntegrator(wave, runge_kutta, dyn)
@time sol = integrate(iter, u0, 1500)
# @time render!(sol, path = "vid.mp4")

using GLMakie

fig = Figure()
ax = Axis3(fig[1, 1], perspectiveness = 0.5, aspect = (1, 1, 1))
zlims!(ax, -1.0, 4.0)

record(fig, "vid.mp4", 1:length(sol)) do i
    empty!(ax.scene)
    surface!(ax, dim.x, dim.y, sol.u[i][:, :, 1], colormap = :ice)
end