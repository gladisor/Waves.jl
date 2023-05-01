export RandomRadiiScattererGrid, RandomRadiiScattererRing, RandomCloak
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
    ring::RandomRadiiScattererRing
    core::Scatterers
end

function (cloak::RandomCloak)()
    return Cloak(cloak.ring(), cloak.core)
end

function design_space(cloak::RandomCloak, scale::Float32)
    return design_space(cloak.ring, scale)
end