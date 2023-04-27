export RandomRadiiScattererGrid

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