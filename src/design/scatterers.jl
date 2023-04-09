export Scatterers, random_pos, radii_design_space, scatterer_formation, random_radii_scatterer_formation

const MIN_RADII = 0.1f0
const MAX_RADII = 1.0f0
const MIN_SPEED = 0.0f0

struct Scatterers <: AbstractDesign
    pos::AbstractMatrix{Float32}
    r::AbstractVector{Float32}
    c::AbstractVector{Float32}
end

Flux.@functor Scatterers

function random_pos(r::AbstractVector{Float32}, disk_r::Float32)
    r = rand.(Uniform.(0.0f0, disk_r .- r))
    θ = rand.(Uniform.(zeros(Float32, size(r)), ones(Float32, size(r)) * 2π))
    pos = hcat(r .* cos.(θ), r .* sin.(θ))
    return pos
end

function Scatterers(;M::Int, r::Float32, disk_r::Float32, c::Float32)
    r = ones(Float32, M) * r
    pos = random_pos(r, disk_r)
    c = ones(Float32, M) * c
    return Scatterers(pos, r, c)
end

function Base.:+(config1::Scatterers, config2::Scatterers)
    # r = clamp.(config1.r .+ config2.r, MIN_RADII, MAX_RADII)
    # c = max.(config1.c .+ config2.c, MIN_SPEED)
    r = max.(0.5f0, config1.r .+ config2.r)
    return Scatterers(config1.pos .+ config2.pos, r, config1.c .+ config2.c)
end

function Base.:-(config1::Scatterers, config2::Scatterers)
    r = config1.r .- config2.r
    c = config1.c .- config2.c
    return Scatterers(config1.pos .- config2.pos, r, c)
end

function Base.:*(config::Scatterers, n::AbstractFloat)
    return Scatterers(config.pos * n, config.r * n, config.c * n)
end

function Base.:*(n::AbstractFloat, config::Scatterers)
    return config * n
end

function Base.:/(config::Scatterers, n::AbstractFloat)
    return Scatterers(config.pos / n, config.r / n, config.c / n)
end

function Base.zero(config::Scatterers)
    return Scatterers(
        zeros(Float32, size(config.pos)), 
        zeros(Float32, size(config.r)), 
        zeros(Float32, size(config.c)))
end

function location_mask(config::Scatterers, g::AbstractArray{Float32, 3})
    pos = config.pos'
    pos = reshape(pos, 1, 1, size(pos)...)
    mask = dropdims(sum((g .- pos) .^ 2, dims = 3), dims = 3) .< reshape(config.r, 1, 1, length(config.r)) .^ 2
    return mask
end

function speed(config::Scatterers, g::AbstractArray{Float32, 3}, ambient_speed::Float32)
    mask = location_mask(config, g)
    ambient_mask = dropdims(sum(mask, dims = 3), dims = 3) .== 0
    C0 = ambient_mask * ambient_speed
    C_design = dropdims(sum(mask .* reshape(config.c, 1, 1, length(config.c)), dims = 3), dims = 3)
    return C0 .+ C_design
end

function CairoMakie.mesh!(ax::Axis, config::Scatterers)
    for i ∈ axes(config.pos, 1)
        mesh!(ax, Circle(Point(config.pos[i, :]...), config.r[i]), color = :gray)
    end
end

function design_space(config::Scatterers, scale::Float32)
    pos_low = ones(Float32, size(config.pos)) * -scale
    pos_high = ones(Float32, size(config.pos)) * scale
    r = zeros(Float32, size(config.r))
    c = zeros(Float32, size(config.c))
    return Scatterers(pos_low, r, c)..Scatterers(pos_high, r, c)
end

function radii_design_space(config::Scatterers, scale::Float32)

    pos = zeros(Float32, size(config.pos))

    radii_low = - scale * ones(Float32, size(config.r))
    radii_high =  scale * ones(Float32, size(config.r))
    c = zeros(Float32, size(config.c))

    return Scatterers(pos, radii_low, c)..Scatterers(pos, radii_high, c)
end

function Base.rand(config::ClosedInterval{Scatterers})
    if all(config.left.pos .< config.right.pos)
        pos = rand.(Uniform.(config.left.pos, config.right.pos))
    else
        pos = config.left.pos
    end

    if all(config.left.r .< config.right.r)
        r = rand.(Uniform.(config.left.r, config.right.r))
    else
        r = zeros(Float32, size(config.left.r))
    end

    if all(config.left.c .< config.right.c)
        c = rand.(Uniform.(config.left.c, config.right.c))
    else
        c = zeros(Float32, size(config.left.c))
    end
    return Scatterers(pos, r, c)
end

function Base.vec(config::Scatterers)
    return vcat(vec(config.pos), config.r, config.c)
end

function Base.display(design::Scatterers)
    println(typeof(design))
end

function scatterer_formation(;width::Int, hight::Int, spacing::Float32, r::Float32, c::Float32, center::Vector{Float32})
    pos = []

    for i ∈ 1:width
        for j ∈ 1:hight
            point = [(i - 1) * (2 * r + spacing), (j - 1) * (2 * r + spacing)]
            push!(pos, point)
        end
    end

    pos = hcat(pos...)'
    pos = (pos .- mean(pos, dims = 1)) .+ center'

    r = ones(Float32, size(pos, 1)) * r
    c = ones(Float32, size(pos, 1)) * c

    return Scatterers(pos, r, c)
end

function random_radii_scatterer_formation(;kwargs...)
    config = scatterer_formation(r = MAX_RADII; kwargs...)
    r = rand(Float32, size(config.r))
    r = r * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    return Scatterers(config.pos, r, config.c)
end