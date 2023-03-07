export WaveEncoder

struct WaveEncoder
    cell::WaveCell
    dynamics::WaveDynamics
    layers::Chain
end

Flux.@functor WaveEncoder
Flux.trainable(encoder::WaveEncoder) = (encoder.layers,)

function WaveEncoder(;
        wave_fields::Int,
        h_fields::Int,
        latent_fields::Int,
        activation::Function,
        wave_dim::TwoDim,
        latent_dim::OneDim,
        cell::WaveCell,
        dynamics::WaveDynamics)

    reduced_size = prod(Int.(size(wave_dim) ./ (2 ^ 3)))
    latent_elements = size(latent_dim)[1]

    encoder_layers = Chain(
        Conv((3, 3), wave_fields => h_fields, activation, pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        MaxPool((2, 2)),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        MaxPool((2, 2)),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation,    pad = SamePad()),
        Conv((3, 3), h_fields => 1, activation,           pad = SamePad()),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(reduced_size, latent_elements * latent_fields, activation),
        z -> reshape(z, latent_elements, latent_fields))

    return WaveEncoder(cell, dynamics, encoder_layers)
end

function (encoder::WaveEncoder)(wave::AbstractArray{Float32, 3}, steps::Int)
    z = encoder.layers(Flux.batch([wave]))
    latents = integrate(encoder.cell, z, encoder.dynamics, steps)
    latents = Flux.flatten(cat(latents..., dims = 3))
    return latents
end

function (encoder::WaveEncoder)(sol::WaveSol{TwoDim}, steps::Int)
    return encoder(first(sol.u), steps)
end