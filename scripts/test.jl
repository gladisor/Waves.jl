using CairoMakie
using SparseArrays
using Flux
using Flux: pullback, mean
using ChainRulesCore
using ChainRulesCore: canonicalize
using Optimisers
using Waves

function plot_wave(dim::OneDim, wave::AbstractVector{Float32}; ylims::Tuple = (-1.0f0, 1.0f0))
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0f0)
    xlims!(ax, dim.x[1], dim.x[end])
    ylims!(ax, ylims...)
    lines!(ax, dim.x, wave)
    return fig, ax
end

function plot_wave(dim::OneDim, wave::AbstractMatrix{Float32}; kwargs...)
    return plot_wave(dim, wave[:, 1]; kwargs...)
end

function plot_wave(dim::TwoDim, wave::AbstractMatrix{Float32})
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1.0)
    heatmap!(ax, dim.x, dim.y, wave, colormap = :ice)
    return fig, ax
end

function plot_wave(dim::TwoDim, wave::AbstractArray{Float32, 3})
    return plot_wave(dim, wave[:, :, 1])
end

function render!(dim::OneDim, u::AbstractArray{Float32, 3}; path::String)
    fig, ax = plot_wave(dim, u[:, :, 1])

    record(fig, path, axes(u, 3), framerate = 60) do i
        empty!(ax)
        lines!(ax, dim.x, u[:, 1, i], color = :blue)
    end
end

function render!(dim::TwoDim, u::AbstractArray{Float32, 4}; path::String)
    
    fig, ax = plot_wave(dim, u[:, :, :, 1])

    record(fig, path, axes(u, 4), framerate = 100) do i
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
    grad::AbstractMatrix
    ambient_speed::Float32
    C::AbstractArray
    bc::AbstractArray
    pml_scale::Float32
    pml::AbstractArray
end

Flux.@functor AcousticWaveDynamics
Flux.trainable(dyn::AcousticWaveDynamics) = (;dyn.pml)

function (dyn::AcousticWaveDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    ∇ = dyn.grad
    C = dyn.C .* dyn.ambient_speed
    b = C .^ 2
    σ = dyn.pml * dyn.pml_scale

    du = b .* (∇ * v) .- σ .* u
    dv = ∇ * u .- σ .* v

    return hcat(dyn.bc .* du, dv)
end

function (dyn::AcousticWaveDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    ## extract fields from wave
    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    ## get parameters from dynamics
    C = dyn.C
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'

    ## compute relevant gradients
    Vxx = ∇ * Vx
    Vyy = (∇ * Vy')'
    Ux = ∇ * U
    Uy = (∇ * U')'

    ## compute time derivative of fields
    dU = b .* (Vxx .+ Vyy) .+ (Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω)
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    ## concatenate into tensor
    return cat(dyn.bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function adjoint_sensitivity(iter::Integrator, u::A, adj::A) where A <: AbstractArray{Float32, 3}
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)

    a = selectdim(adj, 3, size(adj, 3))
    wave = selectdim(u, 3, size(u, 3))
    _, back = pullback(_iter -> _iter(wave, tspan[end]), iter)

    gs = back(a)[1]
    tangent = Tangent{typeof(iter.dynamics)}(;gs.dynamics...)

    for i in reverse(1:size(u, 3))

        wave = selectdim(u, 3, i)

        adjoint_state = selectdim(adj, 3, i)
        _, back = pullback((_iter, _wave) -> _iter(_wave, tspan[i]), iter, wave)
        
        dparams, dwave = back(adjoint_state)

        a .+= adjoint_state .+ dwave
        tangent += Tangent{typeof(iter.dynamics)}(;dparams.dynamics...)
    end

    return a, tangent
end

function train(iter::Integrator, wave::AbstractMatrix{Float32})

    opt_state = Optimisers.setup(Optimisers.Adam(1e-4), iter)

    for i in 1:100
        z = iter(wave)
        loss, back = pullback(_z -> mean(sum(_z[:, 1, :] .^ 2, dims = 1)), z)
        adj = back(one(loss))[1]
        @time a, tangent = adjoint_sensitivity(iter, z, adj);
        iter_tangent = Tangent{Integrator}(;dynamics = tangent)
        opt_state, iter = Optimisers.update(opt_state, iter, iter_tangent)

        println("Iteration $i, Energy: $loss")
    end

    return iter
end

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00001f0
steps = 200
tf = ti + steps * dt

pulse_intensity = 5.0f0

dim = OneDim(grid_size, elements)
grid = build_grid(dim)
grad = build_gradient(dim)
ambient_speed = 1543.0f0
C = ones(Float32, size(dim)...)
bc = dirichlet(dim)
pml_scale = 15000.0f0
pml = build_pml(dim, 1.0f0, 0.0f0)

dynamics = gpu(AcousticWaveDynamics(grad, ambient_speed, C, bc, pml_scale, pml))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)

pulse = Pulse(dim, 0.0f0, pulse_intensity)
wave = gpu(pulse(build_wave(dim, fields = 2)))

iter = train(iter, wave)
z = iter(wave)

render!(dim, z, path = "vid.mp4")

# C = iter.dynamics.C
# fig, ax = plot_wave(dim, C, ylims = (minimum(C), maximum(C)))
# save("C.png", fig)

pml = iter.dynamics.pml
fig, ax = plot_wave(dim, pml, ylims = (minimum(pml), maximum(pml)))
save("pml.png", fig)