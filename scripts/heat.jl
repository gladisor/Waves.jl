using Waves
using Flux
using CairoMakie

struct HeatDynamics{D <: AbstractDim} <: AbstractDynamics
    grad::AbstractMatrix
end

Flux.@functor HeatDynamics
Flux.trainable(::HeatDynamics) = (;)

function HeatDynamics(dim::AbstractDim)
    return HeatDynamics{typeof(dim)}(build_gradient(dim))
end

function (dyn::HeatDynamics{OneDim})(x::AbstractArray, t::AbstractVector, θ)
    return θ .* (dyn.grad * dyn.grad * x)
end

dim = OneDim(2.0f0 * pi, 128)
x = dim.x
dyn = HeatDynamics(dim)
iter = Integrator(runge_kutta, dyn, 0.0f0, 1f-3, 500)

t = collect(build_tspan(iter))[:, :]
ui = exp.(- dim.x .^ 2)

θ = 10.f0
u = iter(ui, t, θ)





# iter(ui, θ)

# cost, back = Flux.pullback(θ) do _θ
#     u = iter(ui, t, _θ)
#     sum(u .^ 2)
# end

# gs = back(one(cost))[1]

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

CairoMakie.record(fig, "vid.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, dim.x, u[:, i], color = :blue)
end