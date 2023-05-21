include("dependencies.jl")

Base.:*(design::AbstractDesign, n::AbstractFloat) = n * design
Base.:*(n::AbstractFloat, design::AbstractDesign) = design * n
Base.:-(d1::AbstractDesign, d2::AbstractDesign) = d1 + (-1.0f0 * d2)

"""
Basic building block for a simple configuration of cylindrical objects.
The cylinders have position, radii, and wavespeed (c).
"""
struct Cylinders <: AbstractDesign
    pos::AbstractMatrix
    r::AbstractVector
    c::AbstractVector
end

Flux.@functor Cylinders
Base.:*(cylinders::Cylinders, n) = Cylinders(cylinders.pos * n, cylinders.r * n, cylinders.c * n)
Base.:+(c1::Cylinders, c2::Cylinders) = Cylinders(c1.pos .+ c2.pos, c1.r .+ c2.r, c1.c .+ c2.c)
Base.zero(cyls::Cylinders) = Cylinders(cyls.pos * 0.0f0, cyls.r * 0.0f0, cyls.c * 0.0f0)
Base.length(cyls::Cylinders) = length(cyls.r)

function location_mask(cyls::Cylinders, grid::AbstractArray{Float32, 3})
    pos = cyls.pos'
    pos = reshape(pos, 1, 1, size(pos)...)
    r = reshape(cyls.r, 1, 1, length(cyls)) .^ 2
    return dropdims(sum((grid .- pos) .^ 2, dims = 3), dims = 3) .< r
end

function speed(cyls::Cylinders, g::AbstractArray{Float32, 3}, ambient_speed::Float32)
    mask = location_mask(cyls, g)
    ambient_mask = dropdims(sum(mask, dims = 3), dims = 3) .== 0
    C0 = ambient_mask * ambient_speed
    C_design = dropdims(sum(mask .* reshape(cyls.c, 1, 1, length(cyls)), dims = 3), dims = 3)
    return C0 .+ C_design
end

function CairoMakie.mesh!(ax::Axis, cyls::Cylinders)
    for i ∈ axes(cyls.pos, 1)
        mesh!(ax, Circle(Point(cyls.pos[i, :]...), cyls.r[i]), color = :gray)
    end

    return nothing
end

abstract type AbstractScatterers <: AbstractDesign end

function Base.:*(design::S, n::AbstractFloat) where S <: AbstractScatterers
    return S(design.cylinders * n)
end

function Base.:+(d1::S, d2::S) where S <: AbstractScatterers
    return S(d1.cylinders + d2.cylinders)
end

function Base.zero(design::S) where S <: AbstractScatterers
    return S(zero(design.cylinders))
end

Base.length(design::AbstractScatterers) = length(design.cylinders)
speed(design::AbstractScatterers, g::AbstractArray{Float32, 3}, ambient_speed::Float32) = speed(design.cylinders, g, ambient_speed)
CairoMakie.mesh!(ax::Axis, design::AbstractScatterers) = mesh!(ax, design.cylinders)

struct AdjustableRadiiScatterers <: AbstractScatterers
    cylinders::Cylinders
end

Flux.@functor AdjustableRadiiScatterers
Flux.trainable(design::AdjustableRadiiScatterers) = (;design.cylinders.r)
Base.vec(design::AdjustableRadiiScatterers) = design.cylinders.r

struct AdjustablePositionScatterers <: AbstractScatterers
    cylinders::Cylinders
end

Flux.@functor AdjustablePositionScatterers
Flux.trainable(design::AdjustablePositionScatterers) = (;design.cylinders.pos)
Base.vec(design::AdjustablePositionScatterers) = vec(design.cylinders.pos)

struct Cloak <: AbstractDesign
    config::AbstractScatterers
    core::Cylinders
end

Flux.@functor Cloak

Base.vec(cloak::Cloak) = vcat(vec(cloak.config), vec(cloak.core))
Base.:+(cloak::Cloak, action::Cylinders) = Cloak(cloak.config + action, cloak.core)
Base.:+(cloak1::Cloak, cloak2::Cloak) = Cloak(cloak1.config + cloak2.config, cloak1.core)
Base.:*(n::AbstractFloat, cloak::Cloak) = Cloak(n * cloak.config, cloak.core)
Base.zero(cloak::Cloak) = zero(cloak.config)
function CairoMakie.mesh!(ax::Axis, design::Cloak)
    mesh!(ax, design.config)
    mesh!(ax, design.core)
end

struct DesignSpace{D <: AbstractDesign}
    low::D
    high::D
end

function uniform_scalar_sample(l::Float32, r::Float32)
    if l < r
        return rand(Uniform(l, r))
    else
        return l
    end
end

function Base.rand(space::DesignSpace{Cylinders})
    pos = uniform_scalar_sample.(space.low.pos, space.high.pos)
    r = uniform_scalar_sample.(space.low.r, space.high.r)
    c = uniform_scalar_sample.(space.low.c, space.high.c)
    return Cylinders(pos, r, c)
end

function Base.rand(space::DesignSpace{S}) where S <: AbstractScatterers
    cylinders = rand(DesignSpace(space.low.cylinders, space.high.cylinders))
    return S(cylinders)
end

function Base.rand(space::DesignSpace{Cloak})
    config = rand(DesignSpace(space.low.config, space.high.config))
    core = rand(DesignSpace(space.low.core, space.high.core))
    return Cloak(config, core)
end

dim = TwoDim(15.0f0, 512)
grid = build_grid(dim)

pos = hcat([
    [0.0f0, 0.0f0],
    # [5.0f0, 0.0f0],
    # [0.0f0, 5.0f0]
]...)'

r_low = fill(0.2f0, size(pos, 1))
r_high = fill(1.0f0, size(pos, 1))
c = fill(AIR, size(pos, 1))

core = Cylinders([4.0f0, 0.0f0]', [2.0f0], [AIR])
design_low = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_low, c)), core)
design_high = Cloak(AdjustableRadiiScatterers(Cylinders(pos, r_high, c)), core)
space = DesignSpace(design_low, design_high)

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, dim.y[1], dim.y[end])
mesh!(ax, rand(space))
save("design.png", fig)