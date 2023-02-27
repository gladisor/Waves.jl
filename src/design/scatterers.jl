export Scatterers, random_pos

struct Scatterers <: AbstractDesign
    pos::AbstractMatrix{Float32}
    r::AbstractVector{Float32}
    c::AbstractVector{Float32}
end

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
    return Scatterers(config1.pos .+ config2.pos, config1.r .+ config2.r, config1.c .+ config2.c)
end

function Base.:-(config1::Scatterers, config2::Scatterers)
    return Scatterers(config1.pos .- config2.pos, config1.r .- config2.r, config1.c .- config2.c)
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
    mask = Waves.location_mask(config, g)
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

function Flux.gpu(config::Scatterers)
    return Scatterers(gpu(config.pos), gpu(config.r), gpu(config.c))
end

function Flux.cpu(config::Scatterers)
    return Scatterers(cpu(config.pos), cpu(config.r), cpu(config.c))
end

function design_space(config::Scatterers, scale::Float32)
    pos_low = ones(Float32, size(config.pos)) * -scale
    pos_high = ones(Float32, size(config.pos)) * scale
    r = zeros(Float32, size(config.r))
    c = zeros(Float32, size(config.c))
    return Scatterers(pos_low, r, c)..Scatterers(pos_high, r, c)
end

function Base.rand(config::ClosedInterval{Scatterers})
    pos = rand.(Uniform.(config.left.pos, config.right.pos))
    r = zeros(Float32, size(config.left.r))
    c = zeros(Float32, size(config.left.c))
    return Scatterers(pos, r, c)
end