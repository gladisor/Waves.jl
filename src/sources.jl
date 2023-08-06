export NoSource, Source

struct NoSource <: AbstractSource end
(source::NoSource)(t::Float32) = 0.0f0

struct Source <: AbstractSource
    source::AbstractArray{Float32}
    freq::Float32
end

Flux.@functor Source
Flux.trainable(::Source) = (;)

Source(source::AbstractArray{Float32}; freq::Float32) = Source(source, freq)

function (source::Source)(t::Float32)
    return source.source * sin(2.0f0 * pi * t * source.freq)
end