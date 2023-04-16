using CairoMakie
using SparseArrays
using Flux
using Flux: pullback, mean
using ChainRulesCore
using ChainRulesCore: canonicalize
using Optimisers
using Waves

struct AcousticWaveDynamics <: AbstractDynamics
    grad::AbstractMatrix
    ambient_speed::Float32
    C::AbstractArray
    bc::AbstractArray
    pml_scale::Float32
    pml::AbstractArray
end

Flux.@functor AcousticWaveDynamics
Flux.trainable(dyn::AcousticWaveDynamics) = (;dyn.C, dyn.pml)

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

struct ReverseAcousticWaveDynamics <: AbstractDynamics
    grad::AbstractMatrix
    ambient_speed::Float32
    C::AbstractArray
    bc::AbstractArray
    pml_scale::Float32
    pml::AbstractArray
end

Flux.@functor ReverseAcousticWaveDynamics
Flux.trainable(dyn::ReverseAcousticWaveDynamics) = (;dyn.C, dyn.pml)

function (dyn::ReverseAcousticWaveDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    ∇ = dyn.grad
    C = dyn.C .* dyn.ambient_speed
    b = C .^ 2
    σ = dyn.pml * dyn.pml_scale

    du = b .* (∇ * v) .+ σ .* u
    dv = ∇ * u .+ σ .* v

    return hcat(dyn.bc .* du, dv)
end

function train(iter::Integrator, wave::AbstractMatrix{Float32})

    opt_state = Optimisers.setup(Optimisers.Adam(1e-3), iter)

    for i in 1:100
        z = iter(wave)

        z, iter_back = pullback(_iter -> _iter(wave), iter)
        loss, loss_back = pullback(_z -> mean(sum(_z[:, 1, :] .^ 2, dims = 1)), z)

        gs = iter_back(loss_back(one(loss))[1])[1]

        opt_state, iter = Optimisers.update(opt_state, iter, gs)
        println("Iteration $i, Energy: $loss")
    end

    return iter
end

grid_size = 10.f0
elements = 512
ti = 0.0f0
dt = 0.00001f0
steps = 800
tf = ti + steps * dt

pulse_intensity = 5.0f0

dim = OneDim(grid_size, elements)
grid = build_grid(dim)
grad = build_gradient(dim)
ambient_speed = 1543.0f0
C = ones(Float32, size(dim)...)
bc = dirichlet(dim)
pml_scale = 15000.0f0
pml = build_pml(dim, 1.0f0, 1.0f0)

dynamics = gpu(AcousticWaveDynamics(grad, ambient_speed, C, bc, pml_scale, pml))
reverse_dynamics = gpu(ReverseAcousticWaveDynamics(grad, ambient_speed, C, bc, pml_scale, pml))
iter = Integrator(runge_kutta, dynamics, ti, dt, steps)
reverse_iter = Integrator(runge_kutta, reverse_dynamics, tf, -dt, steps)

pulse = Pulse(dim, 0.0f0, pulse_intensity)
wave = gpu(pulse(build_wave(dim, fields = 2)))

iter = train(iter, wave)
@time z = iter(wave)
@time render!(dim, z, path = "z.mp4")
z_reverse = reverse_iter(z[:, :, end])
@time render!(dim, z_reverse, path = "z_reverse.mp4")

C = iter.dynamics.C
fig, ax = plot_wave(dim, C, ylims = (minimum(C), maximum(C)))
save("C.png", fig)

pml = iter.dynamics.pml
fig, ax = plot_wave(dim, pml, ylims = (minimum(pml), maximum(pml)))
save("pml.png", fig)