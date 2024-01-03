export LocalizationLayer, SinWaveEmbedder, WaveInputLayer, TotalWaveInput, ResidualBlock

"""
Layer which adds two channels of coordinates to 2D images.
"""
struct LocalizationLayer
    coords::AbstractArray{Float32, 4}
end

Flux.@functor LocalizationLayer
Flux.trainable(::LocalizationLayer) = (;)

function LocalizationLayer(dim::TwoDim, resolution::Tuple{Int, Int})
    x = imresize(build_grid(dim), resolution) ./ maximum(dim.x)
    return LocalizationLayer(x[:, :, :, :])
end

function (layer::LocalizationLayer)(x)
    return cat(
        x,
        repeat(layer.coords, 1, 1, 1, size(x, 4)),
        dims = 3)
end


"""
Layer which takes in frequency coeficients and outputs functions over 1D space.
"""
struct SinWaveEmbedder
    frequencies::AbstractMatrix{Float32}
end

Flux.@functor SinWaveEmbedder
Flux.trainable(::SinWaveEmbedder) = (;)

function SinWaveEmbedder(dim::OneDim, nfreq::Int)
    dim = cpu(dim)
    L = dim.x[end] - dim.x[1]
    C = L / 2.0f0

    n = Float32.(collect(1:nfreq))
    frequencies = (pi * n .* (dim.x' .- C)) / L
    return SinWaveEmbedder(permutedims(sin.(frequencies), (2, 1)))
end

function (embedder::SinWaveEmbedder)(x::AbstractMatrix{Float32})
    x_norm = x ./ Float32(sqrt(size(embedder.frequencies, 2)))
    y = (embedder.frequencies * x_norm)
    return y
end

function (embedder::SinWaveEmbedder)(x::AbstractArray{Float32, 3})
    x_norm = x ./ Float32(sqrt(size(embedder.frequencies, 2)))
    y = batched_mul(embedder.frequencies, x_norm)
    return y
end


"""
WaveInputLayer is an abstract input layer which handles the conversion from a WaveEnvState
or a vector of WaveEnvState(s) to the correct input format to a CNN model.
"""
abstract type WaveInputLayer end
(input::WaveInputLayer)(s::Vector{WaveEnvState}) = cat(input.(s)..., dims = 4)

struct TotalWaveInput <: WaveInputLayer end
Flux.@functor TotalWaveInput
(input::TotalWaveInput)(s::WaveEnvState) = s.wave[:, :, :, :] .+ 1f-5

struct ResidualBlock
    main::Chain
    skip::Conv
    activation::Function
    pool::MaxPool
end

Flux.@functor ResidualBlock

function ResidualBlock(k::Tuple{Int, Int}, in_channels::Int, out_channels::Int, activation::Function)
    main = Chain(
        Conv(k, in_channels => out_channels, pad = SamePad()),
        activation,
        Conv(k, out_channels => out_channels, pad = SamePad())
    )

    skip = Conv((1, 1), in_channels => out_channels, pad = SamePad())

    return ResidualBlock(main, skip, activation, MaxPool((2, 2)))
end

function (block::ResidualBlock)(x::AbstractArray{Float32})
    return (block.main(x) .+ block.skip(x)) |> block.activation |> block.pool
end