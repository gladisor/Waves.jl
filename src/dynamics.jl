export
    build_tspan, runge_kutta, euler,
    Integrator, SplitWavePMLDynamics, update_design,
    dirichlet, adjoint_sensitivity, continuous_backprop

function build_tspan(ti::Float32, dt::Float32, steps::Int)
    return collect(range(ti, ti + steps * dt, steps + 1))
end

#=
Defines the runge_kutta integration function
=#
function runge_kutta(f::AbstractDynamics, u::AbstractArray{Float32}, t::Float32, dt::Float32)
    k1 = f(u, t)
    k2 = f(u .+ 0.5f0 * dt * k1, t + 0.5f0 * dt)
    k3 = f(u .+ 0.5f0 * dt * k2, t + 0.5f0 * dt)
    k4 = f(u .+ dt * k3,         t + dt)
    du = 1/6f0 * (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
    return du * dt
end

function euler(f::AbstractDynamics, u::AbstractArray{Float32}, t::Float32, dt::Float32)
    return f(u, t) * dt
end

struct Integrator
    integration_function::Function
    dynamics::AbstractDynamics
    ti::Float32
    dt::Float32
    steps::Int
end

Flux.@functor Integrator
Flux.trainable(iter::Integrator) = (;iter.dynamics,)

function (iter::Integrator)(u::AbstractArray{Float32}, t::Float32)
    return iter.integration_function(iter.dynamics, u, t, iter.dt)
end

function emit(iter::Integrator, u::AbstractArray{Float32}, t::Float32)
    u′ = u .+ iter(u, t)
    return (u′, u′)
end

function (iter::Integrator)(ui::AbstractArray{Float32})
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)[1:end - 1]
    recur = Recur((_u, _t) -> emit(iter, _u, _t), ui)
    return cat(ui, [recur(t) for t in tspan]..., dims = ndims(ui) + 1)
end

function Base.reverse(iter::Integrator)
    tf = iter.ti + iter.steps * iter.dt
    return Integrator(iter.integration_function, iter.dynamics, tf, -iter.dt, iter.steps)
end

# function continuous_backprop(iter::Integrator, u::AbstractArray{Float32, 3}, adj::AbstractArray{Float32, 3}, θ::Params)
#     println("calling continuous_backprop")
#     ## create timespan and a reversed iterator
#     tspan = build_tspan(iter.ti, iter.dt, iter.steps)
#     ## setting the wave to not mutate the original data
#     wave = u[:, :, end]
#     ## initializing an array with the final adjoint state
#     gs = [adj[:, :, end]]
#     _, back = pullback(() -> iter(wave, tspan[end]), θ)
#     θ_gs = back(adj[:, :, end])

#     for i in reverse(axes(tspan, 1))
#         wave = u[:, :, i]
        
#         ## computing sensitivity of dynamics to the current state
#         _, back = pullback(_wave -> iter(_wave, tspan[i]), wave)
#         a = adj[:, :, i] .+ back(adj[:, :, i])[1]
#         push!(gs, a)

#         ## computing the sensitivity of dynamics to the implicit parameters
#         _, back = pullback(() -> iter(wave, tspan[i]), θ)

#         ## accumulating parameter gradients
#         θ_gs .+= back(adj[:, :, i])
#     end

#     ## summing the intermediate adjoint states over the time dimension
#     adj_0 = dropdims(sum(batch(gs), dims = 3), dims = 3)
#     return (θ_gs, adj_0)
# end

function adjoint_sensitivity(iter::Integrator, u::A, adj::A) where A <: AbstractArray{Float32, 3}
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)

    a = adj[:, :, end]
    wave = u[:, :, end]
    _, back = pullback(_iter -> _iter(wave, tspan[end]), iter)

    gs = back(a)[1]
    tangent = Tangent{typeof(iter.dynamics)}(;gs.dynamics...)

    for i in reverse(1:size(u, 3))

        wave = u[:, :, i]
        adjoint_state = adj[:, :, i]
        _, back = pullback((_iter, _wave) -> _iter(_wave, tspan[i]), iter, wave)
        
        dparams, dwave = back(adjoint_state)

        a .+= adjoint_state .+ dwave
        tangent += Tangent{typeof(iter.dynamics)}(;dparams.dynamics...)
    end

    return a, tangent
end

function Flux.ChainRulesCore.rrule(iter::Integrator, ui::AbstractMatrix{Float32})
    u = iter(ui)

    function Integrator_back(adj::AbstractArray{Float32, 3})
        a, tangent = adjoint_sensitivity(iter, u, adj)
        iter_tangent = Tangent{Integrator}(;dynamics = tangent)
        return iter_tangent, a
    end

    return u, Integrator_back
end

struct SplitWavePMLDynamics{D <: Union{Union{DesignInterpolator, Extrapolation}, Nothing}} <: AbstractDynamics
    design::D
    dim::AbstractDim
    grid::AbstractArray{Float32}
    ambient_speed::Float32
    grad::AbstractArray{Float32}
    bc::AbstractArray{Float32}
    pml::AbstractArray{Float32}
end

Flux.@functor SplitWavePMLDynamics
Flux.trainable(dynamics::SplitWavePMLDynamics) = ()

function Waves.speed(dynamics::SplitWavePMLDynamics{Nothing}, t::Float32)
    return dynamics.ambient_speed
end

function Waves.speed(dynamics::SplitWavePMLDynamics, t::Float32)
    return speed(dynamics.design(t), dynamics.grid, dynamics.ambient_speed)
end

function update_design(dynamics::SplitWavePMLDynamics{DesignInterpolator}, tspan::Vector{Float32}, action::AbstractDesign)::SplitWavePMLDynamics{DesignInterpolator}
    initial = dynamics.design(tspan[1])
    design = DesignInterpolator(initial, action, tspan[1], tspan[end])
    return SplitWavePMLDynamics(design, dynamics.dim, dynamics.grid, dynamics.ambient_speed, dynamics.grad, dynamics.bc, dynamics.pml)
end

function update_design(dynamics::SplitWavePMLDynamics{<: Extrapolation}, tspan::Vector{Float32}, action::AbstractDesign)::SplitWavePMLDynamics{<: Extrapolation}
    initial = dynamics.design(tspan[1])
    final = initial + action
    design = linear_interpolation([tspan[1], tspan[end]], [initial, final], extrapolation_bc = Flat())
    return SplitWavePMLDynamics(design, dynamics.dim, dynamics.grid, dynamics.ambient_speed, dynamics.grad, dynamics.bc, dynamics.pml)
end

function (dyn::SplitWavePMLDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    C = dyn.ambient_speed
    ∇ = dyn.grad
    σ = dyn.pml


    du = C ^ 2 * ∂x(∇, v) .- σ .* u
    dv = ∂x(∇, u) .- σ .* v

    return hcat(dyn.bc .* du, dv)
end

function (dyn::SplitWavePMLDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)

    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    C = speed(dyn, t)
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U)
    Uy = ∂y(∇, U)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dyn.bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function dirichlet(dim::OneDim)
    bc = ones(Float32, size(dim)[1])
    bc[[1, end]] .= 0.0f0
    return bc
end

function dirichlet(dim::TwoDim)
    bc = one(dim)
    bc[:, 1] .= 0.0f0
    bc[1, :] .= 0.0f0
    bc[:, end] .= 0.0f0
    bc[end, :] .= 0.0f0
    return bc
end

struct LatentPMLWaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    pml_scale::Float32

    pml::AbstractArray
    grad::AbstractMatrix
    bc::AbstractArray
end

Flux.@functor LatentPMLWaveDynamics
Flux.trainable(dyn::LatentPMLWaveDynamics) = (;dyn.pml)

function LatentPMLWaveDynamics(dim::AbstractDim; ambient_speed::Float32, pml_scale::Float32)
    # pml = zeros(Float32, size(dim)...)
    pml = sqrt.(build_pml(dim, dim.x[end], 1.0f0))
    grad = build_gradient(dim)
    bc = dirichlet(dim)
    return LatentPMLWaveDynamics(ambient_speed, pml_scale, pml, grad, bc)
end

function (dyn::LatentPMLWaveDynamics)(u::AbstractMatrix{Float32}, t::Float32)
    U = u[:, 1]
    V = u[:, 2]

    b = dyn.ambient_speed .^ 2 * u[:, 3]
    ∇ = dyn.grad
    σ = (dyn.pml .^ 2) * dyn.pml_scale

    du = b .* ∂x(∇, V) .- σ .* U
    dv = ∂x(∇, U) .- σ .* V
    dc = b * 0.0f0

    return hcat(dyn.bc .* du, dv, dc)
end