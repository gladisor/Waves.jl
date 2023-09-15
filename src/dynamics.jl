export build_tspan, runge_kutta
export Integrator
export AcousticDynamics

function build_tspan(ti::Float32, dt::Float32, steps::Int)
    return collect(range(ti, ti + steps * dt, steps + 1))
end

function runge_kutta(f::AbstractDynamics, u::AbstractArray{Float32}, t, θ, dt::Float32)
    k1 = f(u,                    t,               θ)
    k2 = f(u .+ 0.5f0 * dt * k1, t .+ 0.5f0 * dt, θ)
    k3 = f(u .+ 0.5f0 * dt * k2, t .+ 0.5f0 * dt, θ)
    k4 = f(u .+ dt * k3,         t .+ dt,         θ)
    du = 1/6f0 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
    return du * dt
end

struct Integrator
    integration_function::Function
    dynamics::AbstractDynamics
    dt::Float32
    # steps::Int
end

Flux.@functor Integrator
Flux.trainable(iter::Integrator) = (;iter.dynamics,)
build_tspan(iter::Integrator, ti::Float32, steps::Int) = build_tspan(ti, iter.dt, steps)

"""
For a batch of predefined time sequences.

Takes in an initial condition ui and a tspan matrix

tspan: (timesteps x batch)

This method is specifically for propagating the dynamics for the predefined number of steps.
"""
function (iter::Integrator)(ui::AbstractArray{Float32}, tspan::AbstractMatrix{Float32}, θ)

    recur = Recur(ui) do _u, _t
        du = iter.integration_function(iter.dynamics, _u, _t, θ, iter.dt)
        u_prime = _u .+ du
        return u_prime, u_prime
    end

    return cat(
        ui, 
        [recur(tspan[i, :]) for i in 1:(size(tspan, 1) - 1)]..., 
        dims = ndims(ui) + 1)
end

function (iter::Integrator)(ui::AbstractArray{Float32}, tspan::AbstractVector{Float32}, θ)
    return iter(ui, tspan[:, :], θ)
end

"""
Major limitation: cannot differentiate through nested parameterized structures.
All parameters must be contained in dynamics.
"""
function adjoint_sensitivity(iter::Integrator, u::A, tspan::AbstractMatrix{Float32}, adj::A) where A <: AbstractArray{Float32}
    a = selectdim(adj, ndims(adj), size(adj, ndims(adj))) ## selecting last timeseries
    a = adj[parentindices(a)...] * 0.0f0 ## getting non-view version @ zero

    wave = selectdim(u, ndims(u), size(u, ndims(u)))
    wave = u[parentindices(wave)...] ## getting non-view version

    _, back = Flux.pullback(_iter -> _iter(wave, tspan[end, :]), iter)

    gs = back(a)[1]
    tangent = Tangent{typeof(iter.dynamics)}(;gs.dynamics...) * 0.0f0 ## starting gradient accumulation at zero

    for i in reverse(1:size(u, ndims(u)))

        wave = selectdim(u, ndims(u), i) ## current wave state
        wave = u[parentindices(wave)...] ## getting non-view version

        adjoint_state = selectdim(adj, ndims(adj), i) ## current adjoint state
        adjoint_state = adj[parentindices(adjoint_state)...] ## getting non-view version

        _, back = Flux.pullback((_iter, _wave) -> _iter(_wave, tspan[i, :]), iter, wave)
        dparams, dwave = back(adjoint_state) ## computing sensitivity of adjoint to params and wave

        a .+= adjoint_state .+ dwave
        tangent += Tangent{typeof(iter.dynamics)}(;dparams.dynamics...)
    end

    return a, tangent
end

"""
Replaces the default autodiff method with a custom adjoint sensitivity method.
"""
# function Flux.ChainRulesCore.rrule(iter::Integrator, ui::AbstractArray{Float32}, t::AbstractMatrix{Float32})

#     println("test")

#     u = iter(ui, t)

#     function Integrator_back(adj::AbstractArray{Float32})
#         a, tangent = adjoint_sensitivity(iter, u, t, adj)
#         iter_tangent = Tangent{Integrator}(;dynamics = tangent)
#         return iter_tangent, a, nothing
#     end

#     return u, Integrator_back
# end

struct AcousticDynamics{D <: AbstractDim} <: AbstractDynamics
    dim::D
    c0::Float32
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor AcousticDynamics
Flux.trainable(::AcousticDynamics) = (;)

function AcousticDynamics(dim::AbstractDim, c0::Float32, pml_width::Float32, pml_scale::Float32)

    return AcousticDynamics(
        dim,
        c0, 
        build_gradient(dim), 
        build_pml(dim, pml_width, pml_scale), 
        build_dirichlet(dim))
end

function acoustic_dynamics(x, c, f, ∇, pml, bc)
    U = x[:, :, 1]
    Vx = x[:, :, 2]
    Vy = x[:, :, 3]
    Ψx = x[:, :, 4]
    Ψy = x[:, :, 5]
    Ω = x[:, :, 6]

    b = c .^ 2

    σx = pml
    σy = σx'

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U .+ f)
    Uy = ∂y(∇, U .+ f)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function (dyn::AcousticDynamics{TwoDim})(x, t::AbstractVector{Float32}, θ)
    C, F = θ

    c = C(t)
    f = F(t)

    dtot = acoustic_dynamics(x[:, :, 1:6],        c, f, dyn.grad, dyn.pml, dyn.bc)
    dinc = acoustic_dynamics(x[:, :, 7:end], dyn.c0, f, dyn.grad, dyn.pml, dyn.bc)
    return cat(dtot, dinc, dims = 3)
end

function (dyn::AcousticDynamics{OneDim})(x::AbstractArray, t::AbstractVector{Float32}, θ)
    C, F = θ

    U_tot = x[:, 1, :]
    V_tot = x[:, 2, :]
    U_inc = x[:, 3, :]
    V_inc = x[:, 4, :]

    ∇ = dyn.grad

    c = C(t)
    f = F(t)

    dU_tot = (dyn.c0 ^ 2 * c) .* (∇ * V_tot) .- dyn.pml .* U_tot
    dV_tot = ∇ * (U_tot .+ f) .- dyn.pml .* V_tot

    dU_inc = (dyn.c0 ^ 2) * (∇ * V_inc) .- dyn.pml .* U_inc
    dV_inc = ∇ * (U_inc .+ f) .- dyn.pml .* V_inc

    return hcat(
        Flux.unsqueeze(dU_tot, 2) .* dyn.bc,
        Flux.unsqueeze(dV_tot, 2),
        Flux.unsqueeze(dU_inc, 2) .* dyn.bc,
        Flux.unsqueeze(dV_inc, 2),
        )
end