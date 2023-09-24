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
Adds two named tuples together preserving fields. Assumes exact same
structure for each.
"""
function add_gradients(gs1::NamedTuple, gs2::NamedTuple)

    v3 = []

    for ((k1, v1), (k2, v2)) in zip(pairs(gs1), pairs(gs2))
        if v1 isa NamedTuple
            push!(v3, Flux.ChainRulesCore.elementwise_add(v1, v2))
        else
            push!(v3, v1 .+ v2)
        end
    end

    return NamedTuple{keys(gs1)}(v3)
end

function add_gradients(gs1::Vector{NamedTuple}, gs2::Vector{NamedTuple})
    return add_gradients.(gs1, gs2)
end

function add_gradients(::Nothing, gs)
    return gs
end

"""
adjoint_sensitivity method specifically for differentiating a batchwise OneDim simulation.

u: (finite elements x fields x batch x time)
t: (time x batch)
adj: same as solution (u)
"""
function adjoint_sensitivity(iter::Integrator, z::AbstractArray{Float32, 4}, t::AbstractMatrix{Float32}, θ, ∂L_∂z::AbstractArray{Float32, 4})
    ∂L_∂z₀ = ∂L_∂z[:, :, :, end] * 0.0f0 ## loss accumulator
    ∂L_∂θ = nothing

    for i in reverse(axes(z, 4))
        zᵢ = z[:, :, :, i]      ## current state
        tᵢ = t[i, :]            ## current time

        _, back = Flux.pullback(zᵢ, θ) do _zᵢ, _θ
            return iter.integration_function(iter.dynamics, _zᵢ, tᵢ, _θ, iter.dt)
        end

        aᵢ = ∂L_∂z[:, :, :, i]  ## gradient of loss wrt zᵢ
        ∂L_∂z₀ .+= aᵢ

        ∂aᵢ_∂tᵢ, ∂aᵢ_∂θ = back(∂L_∂z₀)
        ∂L_∂z₀ .+= ∂aᵢ_∂tᵢ
        ∂L_∂θ = add_gradients(∂L_∂θ, ∂aᵢ_∂θ)
    end

    return ∂L_∂z₀, ∂L_∂θ
end

function Flux.ChainRulesCore.rrule(iter::Integrator, z0::AbstractArray{Float32, 3}, t::AbstractMatrix{Float32}, θ)
    z = iter(z0, t, θ)
    function Integrator_back(adj::AbstractArray{Float32})
        gs_z0, gs_θ = adjoint_sensitivity(iter, z, t, θ, adj)
        return nothing, gs_z0, nothing, gs_θ
    end

    return z, Integrator_back
end

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
    ∇ = dyn.grad

    U_tot = x[:, 1, :]
    V_tot = x[:, 2, :]
    U_inc = x[:, 3, :]
    V_inc = x[:, 4, :]

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