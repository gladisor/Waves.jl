using Flux
Flux.CUDA.allowscalar(false)
using Flux: DataLoader
using Flux.Losses: mse, huber_loss, mae
using CairoMakie
using Statistics: mean
using ProgressMeter: @showprogress
using BSON

using Waves

function total_energy(sol::WaveSol)
    return sum.(energy.(displacement.(sol.u)))
end

function total_scattered_energy(s::WaveEnvState)
    return total_energy(s.sol.scattered)
end

struct SigmaControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    z_cell::WaveCell
    z_dynamics::WaveDynamics
    mlp::Chain
end

Flux.@functor SigmaControlModel (wave_encoder, design_encoder, mlp)

function (model::SigmaControlModel)(sol::WaveSol, design::AbstractDesign, action::AbstractDesign)

    z = hcat(model.wave_encoder(sol), model.design_encoder(design, action))

    latents = cat(
        integrate(model.z_cell, z, model.z_dynamics, length(sol) - 1)..., 
        dims = 3)

    return vec(model.mlp(Flux.flatten(latents)))
end

function (model::SigmaControlModel)(s::WaveEnvState, action::AbstractDesign)
    x = cat(s.sol.total.u[1], s.sol.incident.u[1], dims = 3)
    z = hcat(model.wave_encoder(x), model.design_encoder(design, action))

    latents = cat(
        integrate(model.z_cell, z, model.z_dynamics, length(sol) - 1)..., 
        dims = 3
        )

    return vec(model.mlp(Flux.flatten(latents)))
end

function Waves.reset!(model::SigmaControlModel)
    Waves.reset!(model.z_dynamics)
end

function plot_predicted_sigma!(data::Vector{Tuple{WaveEnvState{Scatterers}, Scatterers}}, model; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Energy: σ", title = "Scattered Energy of Solution over Time")

    s, a = data[1]
    t = s.sol.total.t[2:end]

    sigma_true = total_scattered_energy(s)[2:end]
    sigma_pred = cpu(model(gpu(s), gpu(a)))

    lines!(ax, t, sigma_true, color = :blue, label = "True Energy")
    lines!(ax, t, sigma_pred, color = :orange, label = "Model Predicted Energy")

    for x in data[2:end]
        s, a = x
        t = s.sol.total.t[2:end]
        
        sigma_true = total_scattered_energy(s)[2:end]
        sigma_pred = cpu(model(gpu(s), gpu(a)))

        lines!(ax, t, sigma_true, color = :blue)
        lines!(ax, t, sigma_pred, color = :orange)
    end

    axislegend(ax)
    save(path, fig)
end

# function plot_loss!(loss::Vector{Float32}; path::String)
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     lines!(ax, loss)
#     save(path, fig)
# end

function plot_loss!(train_loss::Vector{Float32}, test_loss::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, train_loss, label = "Train Loss")
    lines!(ax, test_loss, label = "Test Loss")

    axislegend(ax)
    save(path, fig)
end

struct LatentSeparation
    base::SigmaControlModel
    incident_encoder::WaveEncoder
end

Flux.@functor LatentSeparation

function (model::LatentSeparation)(s::WaveEnvState, a::AbstractDesign)

    z_total = hcat(model.base.wave_encoder(s.sol.total), model.base.design_encoder(s.design, a))
    z_incident = model.incident_encoder(s.sol.incident)

    total_latents = cat(integrate(model.base.z_cell, z_total, model.base.z_dynamics, length(s.sol.total) - 1)..., dims = 3)
    incident_latents = cat(integrate(model.base.z_cell, z_incident, model.base.z_dynamics, length(s.sol.incident) - 1)..., dims = 3)

    scattered_latents = total_latents .- incident_latents
    return vec(model.base.mlp(Flux.flatten(scattered_latents)))
end

function Waves.reset!(model::LatentSeparation)
    Waves.reset!(model.base)
end

struct FEMIntegrator
    cell::WaveCell
    dynamics::AbstractDynamics
    steps::Int
end

Flux.@functor FEMIntegrator

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

data1 = load_episode_data.(readdir("data/episode1", join = true))
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
s, a = gpu.(first(data1))

Flux.@functor WaveDynamics
Flux.trainable(dynamics::WaveDynamics) = ()

fields = 6
elements = 256
h_fields = 32
h_size = 128

wave_encoder = WaveEncoder(fields, h_fields, 2, relu) |> gpu
design_encoder = DesignEncoder(2 * length(vec(s.design)), h_size, elements, relu) |> gpu
iter = FEMIntegrator(elements, 100; grid_size = 1.0f0, dynamics_kwargs...) |> gpu
mlp = MLP(3 * elements, h_size, 2, 1, relu) |> gpu

ps = Flux.params(wave_encoder, design_encoder, mlp)

opt = Adam(0.0001)

sigma_true = gpu(total_scattered_energy(s)[2:end])

gs = Flux.gradient(ps) do
    z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
    sigma_pred = mlp(Flux.flatten(iter(z)))
    mse(sigma_true, sigma_pred)
end

Flux.Optimise.update!(opt, ps, gs)

# data2 = load_episode_data.(readdir("data/episode2", join = true))
# data3 = load_episode_data.(readdir("data/episode3", join = true))
# data4 = load_episode_data.(readdir("data/episode4", join = true))
# data = vcat(
#     data1, 
#     data2,
#     data3, 
#     data4
#     )

# val_data = load_episode_data.(readdir("data/episode5", join = true))
# test_data = load_episode_data.(readdir("data/episode6", join = true))

# s, a = first(data)

# dim = s.sol.total.dim
# elements = size(dim)[1]
# grid_size = maximum(dim.x)

# z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))
# fields = size(s.sol.total.u[1], 3)
# h_fields = 64
# z_fields = 2
# activation = relu
# design_size = 2 * length(vec(s.design))
# h_size = 1024

# dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

# wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
# design_encoder = DesignEncoder(design_size, h_size, z_elements, activation)
# cell = WaveCell(nonlinear_latent_wave, runge_kutta)
# dynamics = WaveDynamics(dim = OneDim(grid_size, z_elements); dynamics_kwargs...) |> gpu

# mlp = Chain(
#     Dense(3 * z_elements, h_size, activation), 
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, 1)
#     )

# model = LatentSeparation(
#     SigmaControlModel(wave_encoder, design_encoder, cell, dynamics, mlp),
#     WaveEncoder(fields, h_fields, z_fields + 1, activation, out_activation = sigmoid)
#     ) |> gpu



# # train_loader = DataLoader(data, shuffle = true)

# # for x in train_loader
# #     s, a = gpu.(x[1])
# #     sigma_true = gpu(total_scattered_energy(s)[2:end])

# #     gs = Flux.gradient(ps) do
# #         loss = mse(sigma_true, model(s, a))

# #         Flux.ignore() do
# #             println(loss)
# #         end

# #         return loss
# #     end

# #     Flux.Optimise.update!(opt, ps, gs)
# # end

# # train_loss = Float32[]
# # test_loss = Float32[]

# # path = mkpath("results/latent_separation_data=4/")

# # for i in 1:50
        
# #     epoch_loss = Float32[]
# #     for x in train_loader
# #         Waves.reset!(model)

# #         s, a = gpu.(x[1])
# #         sigma_true = gpu(total_scattered_energy(s)[2:end])

# #         gs = Flux.gradient(ps) do
# #             loss = mse(sigma_true, model(s, a))
# #             Flux.ignore(() -> push!(epoch_loss, loss))
# #             return loss
# #         end

# #         Flux.Optimise.update!(opt, ps, gs)
# #     end

# #     push!(train_loss, mean(epoch_loss))

# #     epoch_loss = Float32[]
# #     for x in test_data
# #         s, a = gpu.(x)
# #         sigma_true = gpu(total_scattered_energy(s)[2:end])
# #         loss = mse(sigma_true, model(s, a))
# #         push!(epoch_loss, loss)
# #     end

# #     push!(test_loss, mean(epoch_loss))

# #     println("Train Loss: $(train_loss[end]), Test Loss: $(test_loss[end])")

# #     if i % 5 == 0
# #         plot_loss!(train_loss, test_loss, path = joinpath(path, "loss.png"))
# #         plot_predicted_sigma!(data, model, path = joinpath(path, "train_sigma.png"))

# #         plot_predicted_sigma!(val_data, model, path = joinpath(path, "val_sigma.png"))
# #         plot_predicted_sigma!(test_data, model, path = joinpath(path, "test_sigma.png"))
# #         BSON.@save joinpath(path, "model.bson") model
# #     end
# # end