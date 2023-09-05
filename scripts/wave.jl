using Waves
using Flux
using CairoMakie

struct AcousticDynamics{D <: AbstractDim} <: AbstractDynamics
    c0::Float32
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor AcousticDynamics
Flux.trainable(::AcousticDynamics) = (;)

function AcousticDynamics(dim::AbstractDim, c0::Float32, pml_width::Float32, pml_scale::Float32)
    
    return AcousticDynamics{typeof(dim)}(
        c0, 
        build_gradient(dim), 
        build_pml(dim, pml_width, pml_scale), 
        build_dirichlet(dim))
end

function (dyn::AcousticDynamics{OneDim})(x, t::AbstractVector{Float32}, θ)
    f = θ

    u = x[:, 1]
    v = x[:, 2]

    du = dyn.c0 .^ 2 * dyn.grad * v .- dyn.pml .* u
    dv = dyn.grad * (u .+ f(t)) .- dyn.pml .* v

    return hcat(du .* dyn.bc, dv)
end

function (dyn::AcousticDynamics{TwoDim})(x, t::AbstractVector{Float32}, θ)

end

function build_normal(x::AbstractVector{Float32}, μ::AbstractVector{Float32}, σ::AbstractVector, a::AbstractVector)

    μ = permutedims(μ)
    σ = permutedims(σ)
    a = permutedims(a)

    f = (1.0f0 ./ (σ * sqrt(2.0f0 * π))) .* a .* exp.(- ((x .- μ) .^ 2) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 2), dims = 2)
end

function build_normal(x::AbstractArray{Float32, 3}, μ::AbstractMatrix, σ::AbstractVector, a::AbstractVector)
    μ = permutedims(μ[:, :, :, :], (3, 4, 2, 1))
    σ = permutedims(σ[:, :, :], (2, 3, 1))
    a = permutedims(a[:, :, :], (2, 3, 1))
    
    f = (1.0f0 ./ (2.0f0 * π * σ.^ 2)) .* a .* exp.(-dropdims(sum((x .- μ) .^ 2, dims = 3), dims = 3) ./ (2.0f0 * σ .^ 2))
    return dropdims(sum(f, dims = 3), dims = 3)
end

dim = TwoDim(10.0f0, 512)
dyn = AcousticDynamics(dim, WATER, 2.0f0, 10000.0f0)
iter = Integrator(runge_kutta, dyn, 1f-5, 500)
tspan = build_tspan(iter, 0.0f0)

# pulse = build_normal(
#     dim.x, 
#     [-5.0f0],
#     [0.3f0],
#     [0.5f0])

grid = build_grid(dim)

n = 3
μ = randn(Float32, n, 2)
σ = ones(Float32, n)
a = ones(Float32, n)

pulse = build_normal(grid, μ, σ, a)

println(sum(pulse) * get_dx(dim) * get_dy(dim))

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
heatmap!(ax, dim.x, dim.y, pulse, colormap = :ice)
save("pulse.png", fig)


# source = Source(pulse, 1000.0f0)

# x = build_wave(dim, 2)

# cost, back = Flux.pullback(source) do _source
#     sol = iter(x, tspan, _source)
#     sum(sol[:, 1, :] .^ 2)
# end

# gs = back(one(cost))[1]

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, dim.x, gs.shape)
# save("gs.png", fig)

# @time sol = iter(x, tspan, source)

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, dim.x[1], dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# CairoMakie.record(fig, "vid.mp4", axes(sol, 3)) do i
#     empty!(ax)
#     lines!(ax, dim.x, sol[:, 1, i], color = :blue)
# end