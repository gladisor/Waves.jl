export NoSource, Source

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