using Waves
using Flux
using CairoMakie

function build_wave_encoder(;
            k::Tuple{Int, Int},
            in_channels::Int,
            activation::Function,
            h_size::Int,
            nfields::Int,
            nfreq::Int,
            latent_dim::OneDim)

    return Chain(
        Waves.TotalWaveInput(),
        Waves.ResidualBlock(k, in_channels, 32, activation),
        Waves.ResidualBlock(k, 32, 64, activation),
        Waves.ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        flatten,
        Dense(h_size, nfields * nfreq),
        b -> reshape(b, nfreq, nfields, :),
        Waves.SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, 1, :], x[:, 2, :] ./ WATER, x[:, 3, :], x[:, 4, :] ./ WATER)
    )
end

struct LatentSource <: AbstractSource
    x::AbstractVector{Float32}

    mu::AbstractVector{Float32}
    mu_low::Float32
    mu_high::Float32

    sig::AbstractVector{Float32}
    sig_low::Float32
    sig_high::Float32

    a::AbstractVector{Float32}
    a_low::Float32
    a_high::Float32

    freq::Float32
end

Flux.@functor LatentSource
Flux.trainable(source::LatentSource) = (;source.mu, source.sig, source.a)

function LatentSource(
        dim::OneDim,
        n::Int, 
        mu_low::Float32, 
        mu_high::Float32,
        sig_low::Float32,
        sig_high::Float32,
        a_low::Float32,
        a_high::Float32,
        freq::Float32)

    mu = randn(Float32, n)
    sig = randn(Float32, n)
    a = randn(Float32, n)

    return LatentSource(dim.x, mu, mu_low, mu_high, sig, sig_low, sig_high, a, a_low, a_high, freq)
end

function get_mu(source::LatentSource)
    return sigmoid(source.mu) * (source.mu_high - source.mu_low) .+ source.mu_low
end

function get_sig(source::LatentSource)
    return sigmoid(source.sig) * (source.sig_high - source.sig_low) .+ source.sig_low
end

function get_a(source::LatentSource)
    return sigmoid(source.a) * (source.a_high - source.a_low) .+ source.a_low
end

function (source::LatentSource)(t::AbstractVector{Float32})
    mu = permutedims(get_mu(source))
    sig = permutedims(get_sig(source))
    a = permutedims(get_a(source))

    x = source.x
    
    f = dropdims(
            sum(
                a .* exp.(-(x .- mu) .^ 2 ./ (2.0f0 * sig .^ 2)), 
                dims = 2),
            dims = 2)

    return f .* permutedims(sin.(2.0f0 * pi * source.freq * t))
end

latent_dim = OneDim(15.0f0, 512)
source = gpu(LatentSource(
    latent_dim, 5, 
    latent_dim.x[1] + 5.0f0, latent_dim.x[end] - 5.0f0,
    0.2f0, 1.0f0, 
    0.0f0, 1.0f0, 
    1000.0f0))

t = 0.0f0:1f-5:0.005f0 |> collect
t = gpu(t[:, :])
ti = t[260, :]

cost, back = Flux.pullback(_source -> sum(_source(ti) .^2 ), source)
gs = back(one(cost))[1]

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)

record(fig, "source.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, vec(cpu(source(t[i, :]))), color = :blue)
end
