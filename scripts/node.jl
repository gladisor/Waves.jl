export NODEEnergyModel, NODEDynamics

struct NODEDynamics <: AbstractDynamics 
    re
end

Flux.@functor NODEDynamics
Flux.trainable(::NODEDynamics) = (;)

function (dyn::NODEDynamics)(x::AbstractArray{Float32}, t::AbstractVector{Float32}, θ)
    C, params = θ
    return dyn.re(params)(vcat(x, Flux.unsqueeze(C(t), 2)))
end

struct NODEEnergyModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    dynamics_params::AbstractVector{Float32}
    dx::Float32
end

Flux.@functor NODEEnergyModel

function NODEEnergyModel(env::WaveEnv, activation::Function, h_size::Int, nfreq::Int, latent_dim::OneDim)

    elements = size(latent_dim)[1]

    nframes = size(env.wave, 4) + 1 ## add additional channel for force shape
    fields = 3

    wave_encoder = WaveEncoder(
        build_cnn_base(env, nframes, activation, h_size),
        Chain(Dense(h_size, elements)))

    design_encoder = DesignEncoder(env, h_size, activation, nfreq, latent_dim)

    mlp = Chain(
        Dense(2 * elements, elements, activation),
        Dense(elements, elements, activation),
        Dense(elements, elements, activation),
        Dense(elements, elements)
    )

    params, re = Flux.destructure(mlp)

    dyn = NODEDynamics(re)
    iter = Integrator(runge_kutta, dyn, env.dt)
    return NODEEnergyModel(wave_encoder, design_encoder, iter, params, get_dx(latent_dim))
end

function generate_latent_solution(model::NODEEnergyModel, s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z0 = Flux.unsqueeze(model.wave_encoder(s), 2)
    C = model.design_encoder(s, a, t)
    θ = [C, model.dynamics_params]
    return model.iter(z0, t, θ)
end

function (model::NODEEnergyModel)(s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32})
    z = generate_latent_solution(model, s, a, t)
    return permutedims(dropdims(sum(z .^ 2, dims = 1) * model.dx, dims = (1, 2)), (2, 1))
end

function node_loss(model::NODEEnergyModel, s::Vector{WaveEnvState}, a::Matrix{<: AbstractDesign}, t::AbstractMatrix{Float32}, y::AbstractArray{Float32, 3})
    return Flux.mse(
        model(s, a, t), 
        y[:, 3, :]
    )
end

function make_plots(model::NODEEnergyModel, batch; path::String, samples::Int)

    s, a, t, y = batch
    y_hat = cpu(model(s, a, t))
    y = cpu(y)

    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        Waves.plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    return nothing
end