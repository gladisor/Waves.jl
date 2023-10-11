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

struct WaveReconstructionModel
    energy_model::AcousticEnergyModel
    decoder::Chain
end

Flux.@functor WaveReconstructionModel

function (model::WaveReconstructionModel)(s, a, t, idx)
    nsample = length(idx) ## how many images per batch are we reconstructing
    batchsize = length(s)

    ## unroll latent dynamics
    z = generate_latent_solution(model.energy_model, s, a, t)

    ## grab only total latent field
    u_tot_z = z[:, 1, :, :]
    
    ## place batch dim at end
    u_tot_z = permutedims(u_tot_z, (1, 3, 2))

    ## select only points in time we are going to reconstruct
    u_tot_z_samples = u_tot_z[:, idx, :]

    ## reshape latent solution into 16 x 16 feature maps
    x = reshape(u_tot_z_samples, (16, 16, 4, nsample, batchsize))

    ## flatten the samples into batch dim
    x = reshape(x, (16, 16, 4, nsample * batchsize))

    w_hat = model.decoder(x)

    ## static (128 x 128) image size assumption!!!
    w_hat = reshape(w_hat, (128, 128, 1, nsample, batchsize))
    return compute_latent_energy(z, model.energy_model.dx),  dropdims(w_hat, dims = 3)
end

function WaveReconstructionModel(energy_model::AcousticEnergyModel, activation::Function)

    decoder = Chain(
        Conv((3, 3), 4 => 64, activation, pad = SamePad()),
        Conv((3, 3), 64 => 128, activation, pad = SamePad()),
        Upsample((2, 2)),

        Conv((3, 3), 128 => 128, activation, pad = SamePad()),
        Conv((3, 3), 128 => 128, activation, pad = SamePad()),
        Upsample((2, 2)),

        Conv((3, 3), 128 => 128, activation, pad = SamePad()),
        Conv((3, 3), 128 => 128, activation, pad = SamePad()),
        Upsample((2, 2)),

        Conv((3, 3), 128 => 64, activation, pad = SamePad()),
        Conv((3, 3), 64 => 64, activation, pad = SamePad()),
        Conv((1, 1), 64 => 1, w -> 3.0f0 * tanh.(w)))

    return WaveReconstructionModel(energy_model, decoder)
end

struct WaveReconstructionLoss
    idx::AbstractVector{Int}
    real_dx::Float32
    real_dy::Float32
end

Flux.@functor WaveReconstructionLoss
Flux.trainable(::WaveReconstructionLoss) = (;)

function (loss::WaveReconstructionLoss)(model::WaveReconstructionModel, s, a, t, y, w)
    y_hat, w_hat = model(s, a, t, loss.idx)
    L_recons = Flux.mean(sum((w .- w_hat) .^ 2, dims = (1, 2)) * loss.real_dx * loss.real_dy)
    L_energy = Flux.mse(y_hat, y)
    return L_recons + L_energy
end

function get_reconstruction_indexes(integration_steps::Int, horizon::Int)
    n = integration_steps + 1
    return vec(collect((n - 2 * Waves.FRAMESKIP):Waves.FRAMESKIP:n) .+ integration_steps * collect(0:horizon-1)')
end

function WaveReconstructionLoss(env::WaveEnv, horizon::Int)
    dx = get_dx(env.dim) * length(env.dim.x) / env.resolution[1]
    dy = get_dy(env.dim) * length(env.dim.y) / env.resolution[2]
    idx = get_reconstruction_indexes(env.integration_steps, horizon)
    return WaveReconstructionLoss(idx, dx, dy)
end
