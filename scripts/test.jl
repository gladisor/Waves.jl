using CairoMakie
using SparseArrays
using Flux
using Waves

function plot_wave(dim::TwoDim, wave::AbstractMatrix{Float32})
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0)
    heatmap!(ax, dim.x, dim.y, wave, colormap = :ice)
    return fig, ax
end

function plot_wave(dim::TwoDim, wave::AbstractArray{Float32, 3})
    return plot_wave(dim, wave[:, :, 1])
end

function render!(dim::TwoDim, u::AbstractArray{Float32, 4}; path::String)
    
    fig, ax = plot_wave(dim, u[:, :, :, 1])

    record(fig, path, axes(u, 4), framerate = 60) do i
        empty!(ax)
        heatmap!(ax, dim.x, dim.y, u[:, :, 1, i], colormap = :ice)
    end
end

function Waves.dirichlet(dim::TwoDim)
    bc = one(dim)
    bc[:, 1] .= 0.0f0
    bc[1, :] .= 0.0f0
    bc[:, end] .= 0.0f0
    bc[end, :] .= 0.0f0
    return bc
end

struct AcousticWaveDynamics <: AbstractDynamics
    grad::SparseMatrixCSC
    C::AbstractArray
    bc::AbstractArray
    pml::AbstractArray
end

Flux.@functor AcousticWaveDynamics ()

function (dyn::AcousticWaveDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    C = dyn.C
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'

    Vxx = ∇ * Vx
    Vyy = (∇ * Vy')'
    Ux = ∇ * U
    Uy = (∇ * U')'

    dU = b .* (Vxx .+ Vyy) .+ (Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω)
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dyn.bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

grid_size = 5.f0
elements = 512
ti = 0.0f0
dt = 0.00001f0
steps = 800

ambient_speed = 1543.0f0
pulse_intensity = 1.0f0

dim = TwoDim(grid_size, elements)

grad = build_gradient(dim)
C = ones(Float32, size(dim)...) * ambient_speed
bc = dirichlet(dim)
pml = build_pml(dim, 1.0f0, 210000.0f0)

dynamics = AcousticWaveDynamics(grad, C, bc, pml)
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

pulse = Pulse(dim, -3.0f0, 0.0f0, pulse_intensity)
wave = pulse(build_wave(dim, fields = 6))

@time u = iter(wave)
@time render!(dim, u, path = "vid.mp4")