
struct FEMIntegrator
    cell::WaveCell
    dynamics::AbstractDynamics
    steps::Int
end

Flux.@functor FEMIntegrator
Flux.trainable(iter::FEMIntegrator) = ()

function FEMIntegrator(elements::Int, steps::Int; grid_size::Float32, dynamics_kwargs...)
    cell = WaveCell(nonlinear_latent_wave, runge_kutta)
    dynamics = WaveDynamics(dim = OneDim(grid_size, elements); dynamics_kwargs...)
    return FEMIntegrator(cell, dynamics, steps)
end

function (iter::FEMIntegrator)(z::AbstractMatrix{Float32})
    latents = cat(integrate(iter.cell, z, iter.dynamics, iter.steps)..., dims = 3)
    iter.dynamics.t = 0
    return latents
end

struct MLP
    layers::Chain
end

Flux.@functor MLP

function MLP(in_size::Int, h_size::Int, n_h::Int, out_size::Int, activation::Function)
    input = Dense(in_size, h_size, activation)
    hidden = [Dense(h_size, h_size, activation) for i in 1:n_h]
    output = Dense(h_size, out_size)
    return MLP(Chain(input, hidden..., output))
end

function (model::MLP)(x::AbstractArray{Float32})
    return vec(model.layers(x))
end

struct SigmaControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::FEMIntegrator
    mlp::MLP
end

Flux.@functor SigmaControlModel

function (model::SigmaControlModel)(s::WaveEnvState, action::AbstractDesign)
    z = hcat(model.wave_encoder(s.sol.total), model.design_encoder(s.design, action))
    return model.mlp(Flux.flatten(model.iter(z)))
end

struct LatentSigmaSeparationModel
    total_model::SigmaControlModel
    incident_encoder::WaveEncoder
    incident_iter::FEMIntegrator
end