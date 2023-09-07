export build_tspan, runge_kutta
export Integrator

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
    steps::Int
end

Flux.@functor Integrator
Flux.trainable(iter::Integrator) = (;iter.dynamics,)
build_tspan(iter::Integrator, ti::Float32) = build_tspan(ti, iter.dt, iter.steps)

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
