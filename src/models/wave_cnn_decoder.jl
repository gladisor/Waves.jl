export WaveCNNDecoder

struct WaveCNNDecoder
    layers::Chain
end

Flux.@functor WaveCNNDecoder

function WaveCNNDecoder(;
        wave_fields::Int,
        h_fields::Int,
        latent_fields::Int,
        wave_dim::TwoDim,
        latent_dim::OneDim,
        activation::Function,
        )

    reduced_size = Int.(size(wave_dim) ./ (2 ^ 3))
    latent_elements = size(latent_dim)[1]

    decoder_layers = Chain(
        Dense(latent_elements * latent_fields, 625, activation),
        z -> reshape(z, reduced_size..., 1, :),
        Conv((3, 3), 1 => h_fields, activation,        pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Upsample((2, 2)),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Upsample((2, 2)),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
        Upsample((2, 2)),
        Conv((3, 3), h_fields => wave_fields, activation, pad = SamePad()))
    
    return WaveCNNDecoder(decoder_layers)
end

function (decoder::WaveCNNDecoder)(x::AbstractMatrix{Float32})
    return decoder.layers(x)
end