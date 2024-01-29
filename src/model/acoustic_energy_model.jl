export compute_latent_energy, build_wave_encoder, AcousticEnergyModel, get_parameters_and_initial_condition, generate_latent_solution

"""
Calculates the energy of the latent 1D solution for total, incident, and scattered energy fields.
"""
function compute_latent_energy(z::AbstractArray{Float32, 4}, dx::Float32)
    tot = z[:, 1, :, :]
    inc = z[:, 3, :, :]
    sc = tot .- inc

    tot_energy = sum(tot .^ 2, dims = 1) * dx
    inc_energy = sum(inc .^ 2, dims = 1) * dx
    sc_energy =  sum(sc  .^ 2, dims = 1) * dx
    return permutedims(vcat(tot_energy, inc_energy, sc_energy), (3, 1, 2))
end

struct SinusoidalSource <: AbstractSource
    freq_coefs::AbstractVector
    emb::SinWaveEmbedder
    freq::Float32
end

Flux.@functor SinusoidalSource
Flux.trainable(source::SinusoidalSource) = (;freq_coefs = source.freq_coefs)

function SinusoidalSource(dim::OneDim, nfreq::Int, freq::Float32)
    freq_coefs = randn(Float32, nfreq) ./ Float32(sqrt(nfreq))
    return SinusoidalSource(freq_coefs, SinWaveEmbedder(dim, nfreq), freq)
end

function (source::SinusoidalSource)(t::AbstractVector{Float32})
    f = source.emb(source.freq_coefs[:, :])
    return f .* sin.(2.0f0 * pi * permutedims(t) * source.freq)
end

function build_wave_encoder(;
        env::WaveEnv,
        latent_dim::OneDim,
        h_size::Int,
        nfreq::Int,
        c0::Float32,
        k::Tuple{Int, Int} = (3, 3),
        in_channels::Int = 3,
        activation::Function = leakyrelu)

    nfields = 6

    return Chain(
        TotalWaveInput(),
        LocalizationLayer(env.dim, env.resolution),
        ResidualBlock(k, 2 + in_channels, 32, activation),
        ResidualBlock(k, 32, 64, activation),
        ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Parallel(
            vcat,
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
        ),
        b -> reshape(b, nfreq, nfields, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(
            x[:, [1], :],       # u_tot
            x[:, [2], :], # ./ c0, # v_tot
            x[:, [3], :],       # u_inc
            x[:, [4], :], # ./ c0, # v_inc
            x[:, [5], :],       # f
            x[:, [6], :] .^ 2   # pml
            )
        )
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

function get_parameters_and_initial_condition(model::AcousticEnergyModel, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    x = model.wave_encoder(s)
    z0 = x[:, 1:4, :]
    F = Source(x[:, 5, :], model.F.freq)
    PML = x[:, 6, :]
    C = model.design_encoder(s, a, t)
    θ = [C, F, PML]
    return z0, θ
end

function generate_latent_solution(model::AcousticEnergyModel, s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z0, θ = get_parameters_and_initial_condition(model, s, a, t)
    return model.iter(z0, t, θ)
end

function (model::AcousticEnergyModel)(s::AbstractVector{WaveEnvState}, a::AbstractArray{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = generate_latent_solution(model, s, a, t)
    return compute_latent_energy(z, model.dx)
end

function AcousticEnergyModel(;
        env::WaveEnv, 
        latent_dim::OneDim,
        in_channels::Int,
        h_size::Int, 
        nfreq::Int, 
        pml_width::Float32,
        pml_scale::Float32)

    wave_encoder = Waves.build_wave_encoder(;
        env,
        latent_dim, 
        h_size, 
        nfreq,
        in_channels,
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