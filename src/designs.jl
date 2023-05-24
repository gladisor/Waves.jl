export ALUMINIUM, COPPER, BRASS, AIR, WATER
export DesignSpace, DesignInterpolator
export NoDesign, Cylinders, AdjustableRadiiScatterers, AdjustablePositionScatterers, Cloak
export speed, build_action_space

# https://www.rshydro.co.uk/sound-speeds/
const ALUMINIUM = 3100.0f0
const COPPER = 2260.0f0
const BRASS = 2120.0f0
# https://www.sfu.ca/sonic-studio-webdav/handbook/Speed__Of_Sound.html
const AIR = 344.0f0
const WATER = 1531.0f0

"""
Constrans the upper and lower bound of a design vector space.
Supports modification of a design within an upper and lower bound.

In the future it will also support constraint functions: g1(...) + g2(...) + ... + gn(...)
"""
struct DesignSpace{D <: AbstractDesign}
    low::D
    high::D
end

function (space::DesignSpace)(design::AbstractDesign, action::AbstractDesign)
    return clamp(design + action, space.low, space.high)
end

"""
Defining abstract interface of AbstractDesign
Overriding a scalar multiplication will implement the other direction. Also division.
Implementing addition will automatically implement subtraction
"""
Base.:*(design::AbstractDesign, n::AbstractFloat) = Float32(n) * design
Base.:*(n::AbstractFloat, design::AbstractDesign) = design * Float32(n)
Base.:/(design::AbstractDesign, n::AbstractFloat) = design * (1.0f0/Float32(n))
Base.:-(d1::AbstractDesign, d2::AbstractDesign) = d1 + (-1.0f0 * d2)

struct NoDesign <: AbstractDesign end
Flux.@functor NoDesign
Base.:+(::NoDesign, ::NoDesign) = NoDesign()
Base.:*(::NoDesign, ::Float32) = NoDesign()
Base.zero(d::NoDesign) = d
speed(::NoDesign, ::AbstractArray{Float32}, ambient_speed::Float32) = ambient_speed

"""
Basic building block for a simple configuration of cylindrical objects.
The cylinders have position, radii, and wavespeed (c).
"""
struct Cylinders <: AbstractDesign
    pos::AbstractMatrix{Float32}
    r::AbstractVector{Float32}
    c::AbstractVector{Float32}
end

Flux.@functor Cylinders

"""
Defining minimal set of functions to create a vector space of Cylinders
"""
Base.:*(cylinders::Cylinders, n::Float32) = Cylinders(cylinders.pos * n, cylinders.r * n, cylinders.c * n)
Base.:+(c1::Cylinders, c2::Cylinders) = Cylinders(c1.pos .+ c2.pos, c1.r .+ c2.r, c1.c .+ c2.c)
Base.zero(cyls::Cylinders) = Cylinders(cyls.pos * 0.0f0, cyls.r * 0.0f0, cyls.c * 0.0f0)
Base.length(cyls::Cylinders) = length(cyls.r)
Base.clamp(cyls::Cylinders, low::Cylinders, high::Cylinders) = Cylinders(clamp.(cyls.pos, low.pos, high.pos), clamp.(cyls.r, low.r, high.r), clamp.(cyls.c, low.c, high.c))
Base.vec(cyls::Cylinders) = vcat(vec(cyls.pos), vec(cyls.r), vec(cyls.c))

function build_action_space(cyls::Cylinders, scale::Float32)
    low = Cylinders((cyls.pos .^ 0.0f0) * -scale, (cyls.r .^ 0.0f0) * -scale, (cyls.c .^ 0.0f0) * -scale)
    high = Cylinders((cyls.pos .^ 0.0f0) * scale, (cyls.r .^ 0.0f0) * scale, (cyls.c .^ 0.0f0) * scale)
    return DesignSpace(low, high)
end

"""
Computes a multi channel mask where each channel corresponds to the location mask of one cylinder.
"""
function location_mask(cyls::Cylinders, grid::AbstractArray{Float32, 3})
    pos = cyls.pos'
    pos = reshape(pos, 1, 1, size(pos)...)
    r = reshape(cyls.r, 1, 1, length(cyls)) .^ 2
    return dropdims(sum((grid .- pos) .^ 2, dims = 3), dims = 3) .< r
end

"""
A required function used to create a wavespeed field that is informed by the state of the Cylinder design.
The wavespeed of each cylinder can vary independently.
"""
function speed(cyls::Cylinders, grid::AbstractArray{Float32, 3}, ambient_speed::Float32)
    mask = location_mask(cyls, grid)
    ambient_mask = dropdims(sum(mask, dims = 3), dims = 3) .== 0
    C0 = ambient_mask * ambient_speed
    C_design = dropdims(sum(mask .* reshape(cyls.c, 1, 1, length(cyls)), dims = 3), dims = 3)
    return C0 .+ C_design
end


"""
Plots a cylinder on an Axis
"""
function CairoMakie.mesh!(ax::Axis, cyls::Cylinders)
    for i ∈ axes(cyls.pos, 1)
        mesh!(ax, Circle(Point(cyls.pos[i, :]...), cyls.r[i]), color = :gray)
    end

    return nothing
end

"""
Stacks two cylinder configurations on top of one another. Useful for evaluating the wavespeed field of multiple configurations.
"""
function stack(c1::Cylinders, c2::Cylinders)
    pos = vcat(c1.pos, c2.pos)
    r = vcat(c1.r, c2.r)
    c = vcat(c1.c, c2.c)
    return Cylinders(pos, r, c)
end


"""
Defining an abstract type to allow for two different types of scatterers. 
AdjustableRadiiScatterers have fixed position and adjusable radii.
AdjustablePositionScatterers have adjusable position and fixed radii.

These designs share many methods so they are linked by an abstract type.
"""
function Base.:*(design::S, n::Float32) where S <: AbstractScatterers
    return S(design.cylinders * n)
end

function Base.:+(d1::S, d2::S) where S <: AbstractScatterers
    return S(d1.cylinders + d2.cylinders)
end

function Base.zero(design::S) where S <: AbstractScatterers
    return S(zero(design.cylinders))
end

function Base.clamp(design::S, low::S, high::S) where S <: AbstractScatterers
    return S(clamp(design.cylinders, low.cylinders, high.cylinders))
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

function build_action_space(design::AdjustableRadiiScatterers, scale::Float32)
    s = build_action_space(design.cylinders, scale)
    low = AdjustableRadiiScatterers(Cylinders(s.low.pos * 0.0f0, s.low.r, s.low.c * 0.0f0))
    high = AdjustableRadiiScatterers(Cylinders(s.high.pos * 0.0f0, s.high.r, s.high.c * 0.0f0))
    return DesignSpace(low, high)
end

struct AdjustablePositionScatterers <: AbstractScatterers
    cylinders::Cylinders
end

Flux.@functor AdjustablePositionScatterers
Flux.trainable(design::AdjustablePositionScatterers) = (;design.cylinders.pos)
Base.vec(design::AdjustablePositionScatterers) = vec(design.cylinders.pos)

function build_action_space(design::AdjustablePositionScatterers, scale::Float32)
    s = build_action_space(design.cylinders, scale)
    low = AdjustablePositionScatterers(Cylinders(s.low.pos, s.low.r * 0.0f0, s.low.c * 0.0f0))
    high = AdjustablePositionScatterers(Cylinders(s.high.pos, s.high.r * 0.0f0, s.high.c * 0.0f0))
    return DesignSpace(low, high)
end

struct Cloak <: AbstractDesign
    config::AbstractScatterers
    core::Cylinders
end

Flux.@functor Cloak

Base.vec(cloak::Cloak) = vec(cloak.config) ## assumes core is static
Base.:+(cloak::Cloak, action::AbstractScatterers) = Cloak(cloak.config + action, cloak.core)
Base.:+(c1::Cloak, c2::Cloak) = Cloak(c1.config + c2.config, c1.core + c2.core)
Base.:*(cloak::Cloak, n::AbstractFloat) = Cloak(cloak.config * n, cloak.core * n)

Base.zero(cloak::Cloak) = Cloak(zero(cloak.config), zero(cloak.core))
Base.clamp(cloak::Cloak, low::Cloak, high::Cloak) = Cloak(clamp(cloak.config, low.config, high.config), clamp(cloak.core, low.core, high.core))
build_action_space(cyls::Cloak, scale::Float32) = build_action_space(cyls.config, scale)
speed(cloak::Cloak, grid::AbstractArray{Float32, 3}, ambient_speed::Float32) = speed(stack(cloak.config.cylinders, cloak.core), grid, ambient_speed)

function CairoMakie.mesh!(ax::Axis, design::Cloak)
    mesh!(ax, design.config)
    mesh!(ax, design.core)
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

"""
Manages linear interpolation between two designs over a fixed time period.
"""
struct DesignInterpolator
    initial::AbstractDesign
    final::AbstractDesign
    ti::Float32
    tf::Float32
end

Flux.@functor DesignInterpolator

function DesignInterpolator(initial::AbstractDesign)
    return DesignInterpolator(initial, zero(initial), 0.0f0, 0.0f0)
end

function (interp::DesignInterpolator)(t::Float32)
    Δt = interp.tf - interp.ti
    Δt = ifelse(Δt > 0.0f0, Δt, 1.0f0)
    Δy = interp.final - interp.initial
    return interp.initial + (clamp(t, interp.ti, interp.tf) - interp.ti) * (Δy / Δt)
end

function hexagon_ring(r::Float32)

    pos = Vector{Vector{Float32}}()
    for i in 1:6
        push!(pos, [r * cos((i-1) * 2pi/6.0f0), r * sin((i-1) * 2pi/6.0f0)])
    end

    return Matrix{Float32}(hcat(pos...)')
end

function build_2d_rotation_matrix(theta)
    alpha = theta * pi / 180.0f0

    return [
        cos(alpha) -sin(alpha);
        sin(alpha)  cos(alpha)]
end