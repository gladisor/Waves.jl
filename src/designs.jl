export speed

export NoDesign
export Scatterers, pos_design_space, radii_design_space, random_pos, random_radii_scatterer_formation
export DesignInterpolator
export ALUMINIUM, COPPER, BRASS

const MIN_RADII = 0.5f0
const MAX_RADII = 2.0f0
const MIN_SPEED = 0.0f0

# https://www.rshydro.co.uk/sound-speeds/
const ALUMINIUM = 3100.0f0
const COPPER = 2260.0f0
const BRASS = 2120.0f0

struct DesignInterpolator
    initial::AbstractDesign
    action::AbstractDesign
    ti::Float32
    tf::Float32
end

Flux.@functor DesignInterpolator

function DesignInterpolator(initial::AbstractDesign)
    return DesignInterpolator(initial, zero(initial), 0.0f0, 0.0f0)
end

function (interp::DesignInterpolator)(t::Float32)
    d = (interp.tf - interp.ti)
    t = ifelse(d > 0.0f0, (t - interp.ti) / d, 0.0f0)
    return interp.initial + t * interp.action
end

#### NoDesign Design ####
struct NoDesign <: AbstractDesign end
Flux.@functor NoDesign
Base.:+(d1::NoDesign, ::NoDesign) = d1
Base.:-(d1::NoDesign, ::NoDesign) = d1
Base.:*(d1::NoDesign, n) = d1
Base.:*(n, d1::NoDesign) = d1
Base.:/(d1::NoDesign, n) = d1
Base.zero(d::NoDesign) = d
speed(::NoDesign, ::AbstractArray{Float32}, ambient_speed::Float32) = ambient_speed
########

#### Scatterers Design ####
struct Scatterers <: AbstractDesign
    pos::AbstractMatrix{Float32}
    r::AbstractVector{Float32}
    c::AbstractVector{Float32}
end

Flux.@functor Scatterers
Flux.trainable(config::Scatterers) = (;config.r)

function Base.:*(config::Scatterers, n::AbstractFloat)
    return Scatterers(config.pos * n, config.r * n, config.c * n)
end

function Base.:*(n::AbstractFloat, config::Scatterers)
    return config * n
end

function Base.:+(config1::Scatterers, config2::Scatterers)
    r = clamp.(config1.r .+ config2.r, MIN_RADII, MAX_RADII)
    c = max.(config1.c .+ config2.c, MIN_SPEED)
    return Scatterers(config1.pos .+ config2.pos, r, c)
end

function Base.:-(config1::Scatterers, config2::Scatterers)
    return config1 + config2 * -1.0f0
end

function Base.zero(config::Scatterers)
    return Scatterers(config.pos * 0.0f0, config.r * 0.0f0, config.c * 0.0f0)
end

function location_mask(config::Scatterers, grid::AbstractArray{Float32, 3})
    pos = config.pos'
    pos = reshape(pos, 1, 1, size(pos)...)
    mask = dropdims(sum((grid .- pos) .^ 2, dims = 3), dims = 3) .< reshape(config.r, 1, 1, length(config.r)) .^ 2
    return mask
end

function speed(config::Scatterers, g::AbstractArray{Float32, 3}, ambient_speed::Float32)
    mask = location_mask(config, g)
    ambient_mask = dropdims(sum(mask, dims = 3), dims = 3) .== 0
    C0 = ambient_mask * ambient_speed
    C_design = dropdims(sum(mask .* reshape(config.c, 1, 1, length(config.c)), dims = 3), dims = 3)
    return C0 .+ C_design
end

function Base.vec(config::Scatterers)
    return config.r
end

function pos_design_space(config::Scatterers, scale::Float32)
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

function Base.display(config::Scatterers)
    println(typeof(config), ":")
    println(vec(config))
end

function random_pos(r::AbstractVector{Float32}, disk_r::Float32)
    r = rand.(Uniform.(0.0f0, disk_r .- r))
    theta = rand.(Uniform.(r * 0.0f0, r .^ 0.0f0 * 2pi))
    pos = hcat(r .* cos.(theta), r .* sin.(theta))
    return Float32.(pos)
end

function random_pos(config::Scatterers, disk_r::Float32)
    pos = random_pos(config.r, disk_r)
    return Scatterers(pos, config.r, config.c)
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

########