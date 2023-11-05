export NODEEnergyModel, NODEDynamics, NODEEnergyLoss, MLP

struct MLP
    d1::Dense
    d2::Dense
end

Flux.@functor MLP

function (mlp::MLP)(x::AbstractArray{Float32})
    return x |> mlp.d1 |> mlp.d2
end

struct NODEEnergyModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    dynamics_params::AbstractVector{Float32}
    dx::Float32
end

Flux.@functor NODEEnergyModel

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

struct NODEDynamics <: AbstractDynamics 
    re
end

Flux.@functor NODEDynamics
Flux.trainable(::NODEDynamics) = (;)

function (dyn::NODEDynamics)(x::AbstractArray{Float32}, t::AbstractVector{Float32}, θ)
    C, params = θ
    return dyn.re(params)(vcat(x, Flux.unsqueeze(C(t), 2)))
end

struct NODEEnergyLoss end
Flux.@functor NODEEnergyLoss

function (loss::NODEEnergyLoss)(model::NODEEnergyModel, s, a, t, y)
    y_sc = y[:, 3, :]
    return Flux.mse(model(s, a, t), y_sc)
end

function make_plots(model::NODEEnergyModel, loss_func::NODEEnergyLoss, s, a, t, y; path::String, samples::Int)

    y_hat = cpu(model(s, a, t))
    y = cpu(y)

    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        Waves.plot_predicted_energy(tspan, y[:, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    return nothing
end