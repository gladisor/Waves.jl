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
    # x_norm = x ./ sum(abs, x, dims = 1)
    x_norm = x
    return (embedder.frequencies * x_norm)
end

function (embedder::SinWaveEmbedder)(x::AbstractArray{Float32, 3})
    # x_norm = x ./ sum(abs, x, dims = 1)
    x_norm = x
    # x_norm = Flux.softmax(x, dims = 1)
    return batched_mul(embedder.frequencies, x_norm)
end

struct SinusoidalSource <: AbstractSource
    freq_coefs::AbstractVector
    emb::SinWaveEmbedder
    freq::Float32
end

Flux.@functor SinusoidalSource
Flux.trainable(source::SinusoidalSource) = (;source.freq_coefs)

function SinusoidalSource(dim::OneDim, nfreq::Int, freq::Float32)
    freq_coefs = randn(Float32, nfreq) ./ nfreq
    return SinusoidalSource(freq_coefs, SinWaveEmbedder(dim, nfreq), freq)
end

function (source::SinusoidalSource)(t::AbstractVector{Float32})
    f = source.emb(source.freq_coefs[:, :])
    return f .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
end

struct GaussianSource <: AbstractSource
    x::AbstractArray
    μ::AbstractArray
    σ::AbstractVector
    a::AbstractVector
    freq::Float32
end

Flux.@functor GaussianSource
Flux.trainable(source::GaussianSource) = (;source.μ, source.σ, source.a)
GaussianSource(dim::AbstractDim, args...) = GaussianSource(build_grid(dim), args...)

function (source::GaussianSource)(t::AbstractVector)
    f = build_normal(source.x, source.μ, source.σ, source.a)
    return f .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
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
        BatchNorm(out_channels),
        activation,
        Conv(k, out_channels => out_channels, pad = SamePad())
    )

    skip = Conv((1, 1), in_channels => out_channels, pad = SamePad())

    return ResidualBlock(main, skip, activation, MaxPool((2, 2)))
end

function (block::ResidualBlock)(x::AbstractArray{Float32})
    return (block.main(x) .+ block.skip(x)) |> block.activation |> block.pool
end

function build_wave_encoder(;
        k::Tuple{Int, Int} = (3, 3),
        in_channels::Int = 3,
        activation::Function = leakyrelu,
        h_size::Int = 256,
        nfields::Int = 4,
        nfreq::Int = 50,
        c0::Float32 = WATER,
        latent_dim::OneDim)

    return Chain(
        TotalWaveInput(),
        ResidualBlock(k, in_channels, 32, activation),
        ResidualBlock(k, 32, 64, activation),
        ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Dense(h_size, nfields * nfreq, tanh),
        b -> reshape(b, nfreq, nfields, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0))
end

"""
Normalizes the design parameter vector between -1 and 1
"""
function normalize(design::AbstractDesign, ds::DesignSpace)
    scale = 2.0f0
    return scale * (vec(design) .- vec(ds.low)) ./ (vec(ds.high) .- vec(ds.low) .+ 1f-3) .- (scale / 2.0f0)
end

struct DesignEncoder
    design_space::DesignSpace
    layers::Chain
    integration_steps::Int
end

Flux.@functor DesignEncoder
Flux.trainable(de::DesignEncoder) = (;de.layers)

function (de::DesignEncoder)(d1::Vector{<: AbstractDesign}, a::Vector{<: AbstractDesign})
    d2 = de.design_space.(d1, a)
    return (d2, d2)
end

function (de::DesignEncoder)(s::Vector{WaveEnvState}, a::Matrix{<:AbstractDesign}, t::AbstractMatrix{Float32})
    t_ = t[1:de.integration_steps:end, :]
    d = [si.design for si in s]
    recur = Flux.Recur(de, d)
    design_sequences = hcat(d, [recur(a[i, :]) for i in axes(a, 1)]...)
    x = normalize.(design_sequences, [de.design_space])
    x_batch = cat([hcat(x[i, :]...) for i in axes(x, 1)]..., dims = 3)
    y = de.layers(x_batch)
    return LinearInterpolation(t_, y)
end

# using Flux.ChainRulesCore: Tangent, ZeroTangent
# """
# adjoint_sensitivity method specifically for differentiating a batchwise OneDim simulation.

# u: (finite elements x fields x batch x time)
# t: (time x batch)
# adj: same as solution (u)
# """
# function adjoint_sensitivity(iter::Integrator, z::AbstractArray{Float32, 4}, t::AbstractMatrix{Float32}, θ, ∂L_∂z::AbstractArray{Float32, 4})
#     ∂L_∂z₀ = ∂L_∂z[:, :, :, 1] * 0.0f0 ## loss accumulator
#     # ∂L_∂θ = ZeroTangent()

#     for i in axes(z, 4)
#         zᵢ = z[:, :, :, i]
#         tᵢ = t[i, :]
#         aᵢ = ∂L_∂z[:, :, :, i]

#         _, back = Flux.pullback(zᵢ, θ) do _zᵢ, _θ
#             return iter.integration_function(iter.dynamics, _zᵢ, tᵢ, _θ, iter.dt)
#         end
        
#         ∂aᵢ_∂tᵢ, ∂aᵢ_∂θ = back(∂L_∂z₀)
#         ∂L_∂z₀ .+= aᵢ .+ ∂aᵢ_∂tᵢ
#     end

#     return ∂L_∂z₀
# end