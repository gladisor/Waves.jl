export RandomRadiiScattererGrid, RandomCloak
export RandomRadiiRings, AnnularCloak
export design_space

## Grid
Base.@kwdef struct RandomRadiiScattererGrid <: AbstractInitialDesign
    width::Int
    height::Int
    spacing::Float32
    c::Float32
    center::AbstractVector{Float32}
end

function (reset_design::RandomRadiiScattererGrid)()
    config = scatterer_formation(
        width = reset_design.width,
        height = reset_design.height,
        spacing = reset_design.spacing,
        r = Waves.MAX_RADII,
        c = reset_design.c,
        center = reset_design.center)

    r = rand(Float32, size(config.r))
    r = r * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    return Scatterers(config.pos, r, config.c)
end

function design_space(reset_design::RandomRadiiScattererGrid, scale::Float32)
    return radii_design_space(reset_design(), scale)
end

function ring_points(ring_radius::Float32, spacing::Float32, n::Int)::AbstractMatrix{Float32}
    R = ring_radius
    θ = 2 * asin((2 * Waves.MAX_RADII + spacing) / (2 * R))

    pos = [[R * cos(pi), R * sin(pi)]]
    for i in 1:n
        push!(pos, [R * cos(pi - i * θ), R * sin(pi - i * θ)])
        push!(pos, [R * cos(pi + i * θ), R * sin(pi + i * θ)])
    end
    push!(pos, [R * cos(0.0f0), R * sin(0.0f0)])

    return hcat(pos...)'
end

## Ring
struct RandomRadiiScattererRing <: AbstractInitialDesign
    ring_radius::Float32
    spacing::Float32
    n::Int
    c::Float32
    center::Vector{Float32}
end

function (reset_design::RandomRadiiScattererRing)()
    pos = ring_points(reset_design.ring_radius, reset_design.spacing, reset_design.n)
    r = rand(Float32, size(pos, 1)) * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    c = fill(reset_design.c, size(pos, 1))
    return Scatterers(pos, r, c)
end

function design_space(reset_design::RandomRadiiScattererRing, scale::Float32)
    return radii_design_space(reset_design(), scale)
end

struct RandomCloak <: AbstractInitialDesign
    config::Union{RandomRadiiScattererRing, RandomRadiiScattererGrid}
    core::Scatterers
end

function (cloak::RandomCloak)()
    return Cloak(cloak.config(), cloak.core)
end

function design_space(cloak::RandomCloak, scale::Float32)
    return design_space(cloak.config, scale)
end

function build_rings(R::Vector{Float32}, N::Vector{Int})
    points = Vector{Vector{Float32}}([])

    for i in axes(R, 1)
        for j in 1:N[i]
            t = j*(2*pi/N[i])
			x = R[i] * cos(t)
			y = R[i] * sin(t)

            push!(points, [x, y])
        end
    end

    return hcat(points...)'
end

struct RandomRadiiRings <: AbstractInitialDesign
    R::Vector{Float32}
    N::Vector{Int}
    C::Float32
end

function (rings::RandomRadiiRings)()
    pos = build_rings(rings.R, rings.N)
    M = size(pos, 1)
    r = rand(Float32, M) * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    config = Scatterers(pos, r, fill(BRASS, M))
    return config
end

function design_space(rings::RandomRadiiRings, scale::Float32)
    return radii_design_space(rings(), scale)
end

struct AnnularCloak <: AbstractInitialDesign
    rings::RandomRadiiRings
    core::Scatterers
end

function (cloak::AnnularCloak)()
    return Cloak(cloak.rings(), cloak.core)
end

function design_space(cloak::AnnularCloak, scale::Float32)
    return design_space(cloak.rings, scale)
end