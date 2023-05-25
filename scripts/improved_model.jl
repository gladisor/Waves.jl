abstract type WaveInputLayer end

struct TotalWaveInput <: WaveInputLayer end
Flux.@functor TotalWaveInput
(input::TotalWaveInput)(s::WaveEnvState) = s.wave_total[:, :, :, :]

struct ScatteredWaveInput <: WaveInputLayer end
Flux.@functor ScatteredWaveInput
(input::ScatteredWaveInput)(s::WaveEnvState) = s.wave_total[:, :, :, :] .- s.wave_incident[:, :, :, :]

struct ResidualBlock
    main::Chain
    skip::Conv
    activation::Function
    pool::MaxPool
end

Flux.@functor ResidualBlock

function ResidualBlock(k::Tuple{Int, Int}, in_channels::Int, out_channels::Int, activation::Function)
    main = Chain(
        Conv(k, in_channels => out_channels, activation, pad = SamePad()),
        Conv(k, out_channels => out_channels, pad = SamePad())
    )

    skip = Conv((1, 1), in_channels => out_channels, pad = SamePad())

    return ResidualBlock(main, skip, activation, MaxPool((2, 2)))
end

function (block::ResidualBlock)(x::AbstractArray{Float32})
    return (block.main(x) .+ block.skip(x)) |> block.activation |> block.pool
end

struct NormalizedDense
    dense::Dense
    norm::LayerNorm
    act::Function
end

function NormalizedDense(in_size::Int, out_size::Int, act::Function)
    return NormalizedDense(Dense(in_size, out_size), LayerNorm(out_size), act)
end

Flux.@functor NormalizedDense

function (dense::NormalizedDense)(x)
    return x |> dense.dense |> dense.norm |> dense.act
end

struct FrequencyDomain
    domain::AbstractMatrix{Float32}
end

Flux.@functor FrequencyDomain
Flux.trainable(::FrequencyDomain) = (;)

function FrequencyDomain(dim::OneDim, nfreq::Int)
    dim = cpu(dim)
    L = dim.x[end] - dim.x[1]
    frequencies = (Float32.(collect(1:nfreq)) .* dim.x') / L
    domain = vcat(sin.(2.0f0 * pi * frequencies), cos.(2.0f0 * pi * frequencies))
    return FrequencyDomain(domain)
end

function (freq::FrequencyDomain)(m)
    return m(freq.domain)
end

function build_mlp(in_size::Int, h_size::Int, n_h::Int, activation::Function)

    return Chain(
        NormalizedDense(in_size, h_size, activation),
        [NormalizedDense(h_size, h_size, activation) for _ in 1:n_h]...)
end


function build_hypernet_wave_encoder(;
        latent_dim::OneDim,
        nfreq::Int,
        h_size::Int,
        activation::Function,
        input_layer::WaveInputLayer,
        )

    embedder = Chain(
        build_mlp(2 * nfreq, h_size, 2, activation), 
        Dense(h_size, 3, tanh)
        )

    ps, re = destructure(embedder)

    model = Chain(
        input_layer,
        MaxPool((4, 4)),
        ResidualBlock((3, 3), 1, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, 128, activation),
        GlobalMaxPool(),
        flatten,
        NormalizedDense(128, 512, activation),
        Dense(512, length(ps), bias = false),
        vec,
        re,
        FrequencyDomain(latent_dim, nfreq),
        Scale([1.0f0, 1.0f0/WATER, 1.0f0], false),
        z -> permutedims(z, (2, 1))
        )

    return model
end

struct HypernetDesignEncoder
    design_space::DesignSpace
    action_space::DesignSpace
    layers::Chain
end

Flux.@functor HypernetDesignEncoder

function HypernetDesignEncoder(
        design_space::DesignSpace,
        action_space::DesignSpace,
        nfreq::Int,
        h_size::Int,
        n_h::Int,
        activation::Function,
        latent_dim::OneDim)

    embedder = Chain(
        build_mlp(2 * nfreq, h_size, 2, activation),
        Dense(h_size, 1, sigmoid))

    ps, re = destructure(embedder)

    in_size = length(vec(design_space.low)) + length(vec(action_space.low))

    layers = Chain(
        build_mlp(in_size, h_size, n_h, activation),
        Dense(h_size, length(ps), bias = false),
        re,
        FrequencyDomain(latent_dim, nfreq),
        vec,
        )

    return HypernetDesignEncoder(design_space, action_space, layers)
end

function (model::HypernetDesignEncoder)(d::AbstractDesign, a::AbstractDesign)
    d = (d - model.design_space.low) / (model.design_space.high - model.design_space.low)
    a = (a - model.action_space.low) / (model.action_space.high - model.action_space.low)
    x = vcat(vec(d), vec(a))
    return model.layers(x)
end