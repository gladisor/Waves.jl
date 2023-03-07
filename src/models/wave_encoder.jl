export WaveEncoder

struct WaveEncoder
    cell::WaveCell
    dynamics::WaveDynamics
    layers::Chain
end

Flux.@functor WaveEncoder
Flux.trainable(encoder::WaveEncoder) = (encoder.layers,)

function (encoder::WaveEncoder)(wave::AbstractArray{Float32, 3}, steps::Int)
    z = encoder.layers(Flux.batch([wave]))
    latents = integrate(encoder.cell, z, encoder.dynamics, steps)
    latents = Flux.flatten(cat(latents..., dims = 3))
    return latents
end

function (encoder::WaveEncoder)(sol::WaveSol{TwoDim}, steps::Int)
    return encoder(first(sol.u), steps)
end