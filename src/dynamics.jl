export
    build_tspan, runge_kutta, euler,
    Integrator, SplitWavePMLDynamics, update_design,
    dirichlet, LinearWaveDynamics

function build_tspan(ti::Float32, dt::Float32, steps::Int)::Vector{Float32}
    return range(ti, ti + steps*dt, steps + 1)
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
    tspan = build_tspan(iter.ti, iter.dt, iter.steps - 1)
    recur = Recur((_u, _t) -> emit(iter, _u, _t), ui)
    return cat(ui, [recur(t) for t in tspan]..., dims = ndims(ui) + 1)
end

function Base.reverse(iter::Integrator)
    tf = iter.ti + iter.steps * iter.dt
    return Integrator(iter.integration_function, iter.dynamics, tf, -iter.dt, iter.steps)
end

# function continuous_backprop(iter::Integrator, wave::AbstractMatrix{Float32}, adj::AbstractArray{Float32, 3}, θ::Params)
# end

function continuous_backprop(iter::Integrator, u::AbstractArray{Float32, 3}, adj::AbstractArray{Float32, 3}, θ::Params)
    println("calling continuous_backprop")
    ## create timespan and a reversed iterator
    tspan = build_tspan(iter.ti, iter.dt, iter.steps)
    ## setting the wave to not mutate the original data
    wave = u[:, :, end]
    ## initializing an array with the final adjoint state
    gs = [adj[:, :, end]]
    _, back = pullback(() -> iter(wave, tspan[end]), θ)
    θ_gs = back(adj[:, :, end])

    for i in reverse(axes(tspan, 1))
        wave = u[:, :, i]
        
        ## computing sensitivity of dynamics to the current state
        _, back = pullback(_wave -> iter(_wave, tspan[i]), wave)
        a = adj[:, :, i] .+ back(adj[:, :, i])[1]
        push!(gs, a)

        ## computing the sensitivity of dynamics to the implicit parameters
        _, back = pullback(() -> iter(wave, tspan[i]), θ)

        ## accumulating parameter gradients
        θ_gs .+= back(adj[:, :, i])
    end

    ## summing the intermediate adjoint states over the time dimension
    adj_0 = dropdims(sum(batch(gs), dims = 3), dims = 3)
    return (θ_gs, adj_0)
end

struct SplitWavePMLDynamics{D <: Union{DesignInterpolator, Nothing}} <: AbstractDynamics
    design::D
    dim::AbstractDim
    g::AbstractArray{Float32}
    ambient_speed::Float32
    grad::AbstractArray{Float32}
    pml::AbstractArray{Float32}
end

Flux.@functor SplitWavePMLDynamics
Flux.trainable(dynamics::SplitWavePMLDynamics) = ()

function Waves.speed(dynamics::SplitWavePMLDynamics{Nothing}, t::Float32)
    return dynamics.ambient_speed
end

function Waves.speed(dynamics::SplitWavePMLDynamics{DesignInterpolator}, t::Float32)
    return speed(dynamics.design(t), dynamics.g, dynamics.ambient_speed)
end

function update_design(dynamics::SplitWavePMLDynamics{DesignInterpolator}, tspan::Vector{Float32}, action::AbstractDesign)::SplitWavePMLDynamics{DesignInterpolator}
    initial = dynamics.design(tspan[1])
    design = DesignInterpolator(initial, action, tspan[1], tspan[end])
    return SplitWavePMLDynamics(design, dynamics.dim, dynamics.g, dynamics.ambient_speed, dynamics.grad, dynamics.pml)
end

function (dyn::SplitWavePMLDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    C = dyn.ambient_speed
    ∇ = dyn.grad
    σ = dyn.pml

    du = C ^ 2 * (∇ * v) .- σ .* u
    dv = ∇ * u .- σ .* v

    return hcat(du, dv)
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

function dirichlet(dim::OneDim)
    bc = ones(Float32, size(dim)[1])
    bc[[1, end]] .= 0.0f0
    return bc
end

struct LinearWaveDynamics <: AbstractDynamics
    C::Float32
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor LinearWaveDynamics
Flux.trainable(dynamics::LinearWaveDynamics) = ()

function (dyn::LinearWaveDynamics)(wave::AbstractMatrix{Float32}, t::Float32)
    u = wave[:, 1]
    v = wave[:, 2]

    du =  (dyn.C ^ 2) .* (dyn.grad * v) .* dyn.bc
    dv = dyn.grad * u
    return hcat(du, dv)
end