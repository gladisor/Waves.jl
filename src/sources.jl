export NoSource, Source, RandomPosGaussianSource

function Waves.reset!(source::AbstractSource)
    return nothing
end

struct NoSource <: AbstractSource end
(source::NoSource)(t) = 0.0f0

struct Source <: AbstractSource
    shape::AbstractArray{Float32}
    freq::Float32
end

Flux.@functor Source

function (source::Source)(t::Float32)
    return source.shape * sin(2.0f0 * pi * t * source.freq)
end

function (source::Source)(t::AbstractVector{Float32})
    return source.shape .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
end

mutable struct RandomPosGaussianSource <: AbstractSource
    grid::AbstractArray{Float32, 3}

    μ_low::AbstractMatrix{Float32}
    μ_high::AbstractMatrix{Float32}

    σ::AbstractVector{Float32}
    a::AbstractVector{Float32}

    shape::AbstractMatrix{Float32}
    freq::Float32
end

Flux.@functor RandomPosGaussianSource
Flux.trainable(::RandomPosGaussianSource) = (;)

function Waves.reset!(source::RandomPosGaussianSource)
    ϵ = rand(Float32, size(source.μ_low)...) ## generate some noise

    if Flux.device(source.grid) != Val{:cpu}()
        ϵ = gpu(ϵ)
    end

    μ = (source.μ_high .- source.μ_low) .* ϵ .+ source.μ_low ## scale noise to high and low bounds
    source.shape = build_normal(source.grid, μ, source.σ, source.a) ## build normal distribution shape
    return nothing
end

function RandomPosGaussianSource(
        grid::AbstractArray{Float32, 3},
        μ_low::AbstractMatrix, 
        μ_high::AbstractMatrix, 
        σ::AbstractVector, 
        a::AbstractVector,
        freq::Float32)

    shape = build_normal(grid, μ_high, σ, a)
    source = RandomPosGaussianSource(grid, μ_low, μ_high, σ, a, shape, freq)
    Waves.reset!(source)
    return source
end

function (source::RandomPosGaussianSource)(t::AbstractVector{Float32})
    return source.shape .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
end