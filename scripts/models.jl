
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

# struct LatentSeparation
#     base::SigmaControlModel
#     incident_encoder::WaveEncoder
# end

# Flux.@functor LatentSeparation

# function (model::LatentSeparation)(s::WaveEnvState, a::AbstractDesign)

#     z_total = hcat(model.base.wave_encoder(s.sol.total), model.base.design_encoder(s.design, a))
#     z_incident = model.incident_encoder(s.sol.incident)

#     total_latents = cat(integrate(model.base.z_cell, z_total, model.base.z_dynamics, length(s.sol.total) - 1)..., dims = 3)
#     incident_latents = cat(integrate(model.base.z_cell, z_incident, model.base.z_dynamics, length(s.sol.incident) - 1)..., dims = 3)

#     scattered_latents = total_latents .- incident_latents
#     return vec(model.base.mlp(Flux.flatten(scattered_latents)))
# end

# function Waves.reset!(model::LatentSeparation)
#     Waves.reset!(model.base)
# end

# struct SigmaControlModel
#     wave_encoder::WaveEncoder
#     design_encoder::DesignEncoder
#     z_cell::WaveCell
#     z_dynamics::WaveDynamics
#     mlp::Chain
# end

# Flux.@functor SigmaControlModel (wave_encoder, design_encoder, mlp)

# function (model::SigmaControlModel)(sol::WaveSol, design::AbstractDesign, action::AbstractDesign)

#     z = hcat(model.wave_encoder(sol), model.design_encoder(design, action))

#     latents = cat(
#         integrate(model.z_cell, z, model.z_dynamics, length(sol) - 1)..., 
#         dims = 3)

#     return vec(model.mlp(Flux.flatten(latents)))
# end

# function (model::SigmaControlModel)(s::WaveEnvState, action::AbstractDesign)
#     x = cat(s.sol.total.u[1], s.sol.incident.u[1], dims = 3)
#     z = hcat(model.wave_encoder(x), model.design_encoder(design, action))

#     latents = cat(
#         integrate(model.z_cell, z, model.z_dynamics, length(sol) - 1)..., 
#         dims = 3
#         )

#     return vec(model.mlp(Flux.flatten(latents)))
# end

# function Waves.reset!(model::SigmaControlModel)
#     Waves.reset!(model.z_dynamics)
# end


# train_loss = Float32[]
# test_loss = Float32[]

# path = mkpath("results/latent_separation_data=4/")

# for i in 1:50
        
#     epoch_loss = Float32[]
#     for x in train_loader
#         Waves.reset!(model)

#         s, a = gpu.(x[1])
#         sigma_true = gpu(total_scattered_energy(s)[2:end])

#         gs = Flux.gradient(ps) do
#             loss = mse(sigma_true, model(s, a))
#             Flux.ignore(() -> push!(epoch_loss, loss))
#             return loss
#         end

#         Flux.Optimise.update!(opt, ps, gs)
#     end

#     push!(train_loss, mean(epoch_loss))

#     # epoch_loss = Float32[]
#     # for x in test_data
#     #     s, a = gpu.(x)
#     #     sigma_true = gpu(total_scattered_energy(s)[2:end])
#     #     loss = mse(sigma_true, model(s, a))
#     #     push!(epoch_loss, loss)
#     # end

#     # push!(test_loss, mean(epoch_loss))

#     println("Train Loss: $(train_loss[end])")
#     # println("Train Loss: $(train_loss[end]), Test Loss: $(test_loss[end])")

#     # if i % 5 == 0
#     #     plot_loss!(train_loss, test_loss, path = joinpath(path, "loss.png"))
#     #     plot_predicted_sigma!(data, model, path = joinpath(path, "train_sigma.png"))

#     #     plot_predicted_sigma!(val_data, model, path = joinpath(path, "val_sigma.png"))
#     #     plot_predicted_sigma!(test_data, model, path = joinpath(path, "test_sigma.png"))
#     #     BSON.@save joinpath(path, "model.bson") model
#     # end
# end