export AcousticEnergyModel, SinWaveEmbedder, plot_energy

function compute_latent_energy(z::AbstractArray{Float32, 4}, dx::Float32)
    tot = z[:, 1, :, :]
    inc = z[:, 3, :, :]
    sc = tot .- inc

    tot_energy = sum(tot .^ 2, dims = 1) * dx
    inc_energy = sum(inc .^ 2, dims = 1) * dx
    sc_energy =  sum(sc  .^ 2, dims = 1) * dx
    return permutedims(vcat(tot_energy, inc_energy, sc_energy), (3, 1, 2))
end

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

# struct SinusoidalSource <: AbstractSource
#     freq_coefs::AbstractVector
#     emb::SinWaveEmbedder
#     freq::Float32
# end

# Flux.@functor SinusoidalSource
# Flux.trainable(source::SinusoidalSource) = (;freq_coefs = source.freq_coefs)

# function SinusoidalSource(dim::OneDim, nfreq::Int, freq::Float32)
#     freq_coefs = randn(Float32, nfreq) ./ Float32(sqrt(nfreq))
#     return SinusoidalSource(freq_coefs, SinWaveEmbedder(dim, nfreq), freq)
# end

# function (source::SinusoidalSource)(t::AbstractVector{Float32})
#     f = source.emb(source.freq_coefs[:, :])
#     return f .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
# end

# struct GaussianSource <: AbstractSource
#     x::AbstractArray
#     μ::AbstractArray
#     σ::AbstractVector
#     a::AbstractVector
#     freq::Float32
# end

# Flux.@functor GaussianSource
# Flux.trainable(source::GaussianSource) = (;source.μ, source.σ, source.a)
# GaussianSource(dim::AbstractDim, args...) = GaussianSource(build_grid(dim), args...)

# function (source::GaussianSource)(t::AbstractVector)
#     f = build_normal(source.x, source.μ, exp.(source.σ), source.a)
#     return f .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
# end

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

function build_wave_encoder(;
        latent_dim::OneDim,
        nfields::Int,
        nfreq::Int,
        c0::Float32,
        k::Tuple{Int, Int} = (3, 3),
        in_channels::Int = 3,
        activation::Function = leakyrelu,
        h_size::Int = 256)

    return Chain(
        TotalWaveInput(),
        ResidualBlock(k, in_channels, 32, activation),
        ResidualBlock(k, 32, 64, activation),
        ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Dense(h_size, nfields * nfreq),
        b -> reshape(b, nfreq, nfields, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(
            x[:, [1], :],       # u_tot
            x[:, [2], :] ./ c0, # v_tot
            x[:, [3], :],       # u_inc
            x[:, [4], :] ./ c0, # v_inc
            x[:, [5], :]        # f
            )
        )
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

struct AcousticEnergyModel
    wave_encoder::Chain
    design_encoder::DesignEncoder
    F::AbstractSource
    iter::Integrator
    dx::Float32
end

Flux.@functor AcousticEnergyModel
Flux.trainable(model::AcousticEnergyModel) = (;model.wave_encoder, model.design_encoder, model.F)

function generate_latent_solution(model::AcousticEnergyModel, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    uvf = model.wave_encoder(s)
    
    z0 = uvf[:, 1:4, :]
    F = Source(uvf[:, 5, :], model.F.freq)
    C = model.design_encoder(s, a, t)
    θ = [C, F]
    return model.iter(z0, t, θ)
end

function (model::AcousticEnergyModel)(s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = generate_latent_solution(model, s, a, t)
    return compute_latent_energy(z, model.dx)
end

function AcousticEnergyModel(;
        env::WaveEnv, 
        latent_dim::OneDim, 
        nfreq::Int, 
        h_size::Int,
        pml_width::Float32,
        pml_scale::Float32)

    wave_encoder = build_wave_encoder(;
        latent_dim, 
        h_size, 
        nfreq,
        nfields = 5,
        c0 = env.iter.dynamics.c0)

    mlp = Chain(
        Dense(length(vec(env.design)), h_size, leakyrelu), 
        Dense(h_size, h_size, leakyrelu), 
        Dense(h_size, h_size, leakyrelu),
        Dense(h_size, h_size, leakyrelu), 
        Dense(h_size, nfreq),
        SinWaveEmbedder(latent_dim, nfreq),
        c -> 2.0f0 * sigmoid.(c))

    design_encoder = DesignEncoder(env.design_space, mlp, env.integration_steps)
    F = SinusoidalSource(latent_dim, nfreq, env.source.freq)
    dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, pml_width, pml_scale)
    iter = Integrator(runge_kutta, dyn, env.dt)
    return AcousticEnergyModel(wave_encoder, design_encoder, F, iter, get_dx(latent_dim))
end

function render(dim::OneDim, z::Array{Float32, 3}, t::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    xlims!(ax, dim.x[1], dim.x[end])
    ylims!(ax, -2.0f0, 2.0f0)

    record(fig, path, axes(t, 1)) do i
        empty!(ax)
        lines!(ax, dim.x, z[:, 1, i], color = :blue)
    end
end

function plot_energy(tspan::Vector{Float32}, energy::Matrix{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, tspan, energy[:, 1, 1])
    lines!(ax, tspan, energy[:, 2, 1])
    lines!(ax, tspan, energy[:, 3, 1])
    save(path, fig)
end