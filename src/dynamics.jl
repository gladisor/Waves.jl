export build_tspan, runge_kutta
export Integrator
export WaveDynamics
export ForceLatentDynamics

function build_tspan(ti::Float32, dt::Float32, steps::Int)
    return range(ti, ti + steps * dt, steps + 1)
end

function runge_kutta(f::AbstractDynamics, u::AbstractArray{Float32}, t, dt::Float32)
    k1 = f(u, t)
    k2 = f(u .+ 0.5f0 * dt * k1, t .+ 0.5f0 * dt)
    k3 = f(u .+ 0.5f0 * dt * k2, t .+ 0.5f0 * dt)
    k4 = f(u .+ dt * k3,         t .+ dt)
    du = 1/6f0 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
    return du * dt
end

struct Integrator{D <: AbstractDynamics}
    integration_function::Function
    dynamics::D
    ti::Float32
    dt::Float32
    steps::Int
end

Flux.@functor Integrator
Flux.trainable(iter::Integrator) = (;iter.dynamics,)

build_tspan(iter::Integrator) = build_tspan(iter.ti, iter.dt, iter.steps)

function Base.reverse(iter::Integrator)
    tf = iter.ti + iter.steps * iter.dt
    return Integrator(iter.integration_function, iter.dynamics, tf, -iter.dt, iter.steps)
end

"""
Works with a time (Float32) or a batch of times (Vector)
"""
function (iter::Integrator)(
        u::AbstractArray{Float32}, 
        t::Union{Float32, <: AbstractVector{Float32}}
        )

    return iter.integration_function(iter.dynamics, u, t, iter.dt)
end

function emit(iter::Integrator, u::AbstractArray{Float32}, t::Float32)
    u′ = u .+ iter(u, t)
    return (u′, u′)
end

"""
For a single (shared) starting time
"""
function (iter::Integrator)(ui::AbstractArray{Float32})
    tspan = @ignore_derivatives build_tspan(iter.ti, iter.dt, iter.steps)[1:end - 1]
    recur = Recur((_u, _t) -> emit(iter, _u, _t), ui)
    return cat(ui, [recur(t) for t in tspan]..., dims = ndims(ui) + 1)
end

"""
For a batch of predefined time sequences.

Takes in an initial condition ui and a tspan matrix

tspan: (timesteps x batch)

This method is specifically for propagating the dynamics for the predefined number of steps.
"""
function (iter::Integrator)(ui::AbstractArray{Float32}, tspan::AbstractMatrix{Float32})

    recur = Recur(ui) do _u, _t
        du = iter(_u, _t)
        u_prime = _u .+ du
        return u_prime, u_prime
    end

    return cat(
        ui, 
        [recur(tspan[i, :]) for i in 1:(size(tspan, 1) - 1)]..., 
        dims = ndims(ui) + 1)
end

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

        a .+= adjoint_state .+ dwave# * iter.dt
        tangent += Tangent{typeof(iter.dynamics)}(;dparams.dynamics...)# * iter.dt
    end

    return a, tangent
end


"""
Replaces the default autodiff method with a custom adjoint sensitivity method.
"""
function Flux.ChainRulesCore.rrule(iter::Integrator, ui::AbstractArray{Float32}, t::AbstractMatrix{Float32})

    u = iter(ui, t)

    function Integrator_back(adj::AbstractArray{Float32})
        a, tangent = adjoint_sensitivity(iter, u, t, adj)
        iter_tangent = Tangent{Integrator}(;dynamics = tangent)
        return iter_tangent, a, nothing
    end

    return u, Integrator_back
end

struct WaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    design::DesignInterpolator
    source::AbstractSource
    grid::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor WaveDynamics
Flux.trainable(dyn::WaveDynamics) = (;)

function WaveDynamics(dim::AbstractDim; 
        ambient_speed::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        design::AbstractDesign = NoDesign(),
        source::AbstractSource = NoSource()
        )

    design = DesignInterpolator(design)
    grid = build_grid(dim)
    grad = build_gradient(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    bc = dirichlet(dim)

    return WaveDynamics(ambient_speed, design, source, grid, grad, pml, bc)
end

function (dyn::WaveDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    C = speed(dyn.design(t), dyn.grid, dyn.ambient_speed)
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'
    force = dyn.source(t)

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U .+ force) #.* mask ## Gradient at boundary of design is 0
    Uy = ∂y(∇, U .+ force) #.* mask ##

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dyn.bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function update_design(dyn::WaveDynamics, interp::DesignInterpolator)
    return WaveDynamics(dyn.ambient_speed, interp, dyn.source, dyn.grid, dyn.grad, dyn.pml, dyn.bc)
end