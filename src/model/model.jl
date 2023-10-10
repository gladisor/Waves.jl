function compute_latent_energy(z::AbstractArray{Float32, 4}, dx::Float32)
    tot = z[:, 1, :, :]
    inc = z[:, 3, :, :]
    sc = tot .- inc

    tot_energy = sum(tot .^ 2, dims = 1) * dx
    inc_energy = sum(inc .^ 2, dims = 1) * dx
    sc_energy =  sum(sc  .^ 2, dims = 1) * dx
    return permutedims(vcat(tot_energy, inc_energy, sc_energy), (3, 1, 2))
end

struct AcousticEnergyModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    dx::Float32
    freq::Float32
end

Flux.@functor AcousticEnergyModel

function embed(model::AcousticEnergyModel, s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32})
    x = model.wave_encoder(s)
    z0 = x[:, 1:4, :]
    C = model.design_encoder(s, a, t)
    F = Source(x[:, 5, :], model.freq)
    σₓ = x[:, 6, :]
    θ = [C, F, σₓ]
    return z0, θ
end

function generate_latent_solution(model::AcousticEnergyModel, s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z0, θ = embed(model, s, a, t)
    return model.iter(z0, t, θ)
end

function (model::AcousticEnergyModel)(s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = generate_latent_solution(model, s, a, t)
    return compute_latent_energy(z, model.dx)
end

function render_latent_solution(model::AcousticEnergyModel, s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32}; path::String)
    z0, θ = embed(model, s, a, t)
    z = cpu(model.iter(z0, t, θ))
    C, F, PML = cpu(θ)

    u_tot = z[:, 1, 1, :]
    u_inc = z[:, 3, 1, :]
    u_sc = u_tot .- u_inc
    latent_dim = cpu(model.iter.dynamics.dim)

    fig = Figure()
    ax = Axis(fig[1, 1])
    xlims!(ax, latent_dim.x[1], latent_dim.x[end])
    ylims!(ax, -2.0f0, 2.0f0)

    record(fig, joinpath(path, "u_tot.mp4"), axes(t, 1)) do i
        empty!(ax)
        lines!(ax, latent_dim.x, u_tot[:, i], color = :blue)
        lines!(ax, latent_dim.x, PML[:, 1], color = :red)
        lines!(ax, latent_dim.x, C(cpu(t[i, :]))[:, 1],  color = :green)
    end

    record(fig, joinpath(path, "u_inc.mp4"), axes(t, 1)) do i
        empty!(ax)
        lines!(ax, latent_dim.x, u_inc[:, i], color = :blue)
        lines!(ax, latent_dim.x, PML[:, 1], color = :red)
    end

    record(fig, joinpath(path, "u_sc.mp4"), axes(t, 1)) do i
        empty!(ax)
        lines!(ax, latent_dim.x, u_sc[:, i], color = :blue)
    end
end