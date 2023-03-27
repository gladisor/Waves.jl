using Flux
Flux.CUDA.allowscalar(false)
using Flux: DataLoader
using Flux.Losses: mse, huber_loss, mae
using CairoMakie
using Statistics: mean
using ProgressMeter: @showprogress
using BSON

using Waves

include("design_encoder.jl")

function total_energy(sol::WaveSol)
    return sum.(energy.(displacement.(sol.u)))
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

function plot_predicted_sigma!(data::Vector{Tuple{WaveEnvState{Scatterers}, Scatterers}}, model::SigmaControlModel; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Energy: σ", title = "Scattered Energy of Solution over Time")

    s, a = data[1]
    t = s.sol.total.t[2:end]

    sigma_true = total_energy(s.sol.scattered)[2:end]
    sigma_pred = cpu(model(gpu(s.sol.total), gpu(s.design), gpu(a)))

    lines!(ax, t, sigma_true, color = :blue, label = "True Energy")
    lines!(ax, t, sigma_pred, color = :orange, label = "Model Predicted Energy")

    for x in data[2:end]
        s, a = x
        t = s.sol.total.t[2:end]

        sigma_true = total_energy(s.sol.scattered)[2:end]
        sigma_pred = cpu(model(gpu(s.sol.total), gpu(s.design), gpu(a)))
        lines!(ax, t, sigma_true, color = :blue)
        lines!(ax, t, sigma_pred, color = :orange)
    end

    axislegend(ax)
    save(path, fig)
end

function plot_loss!(loss::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, loss)
    save(path, fig)
end

data1 = load_episode_data.(readdir("data/episode1", join = true))
data2 = load_episode_data.(readdir("data/episode2", join = true))
data3 = load_episode_data.(readdir("data/episode3", join = true))
data4 = load_episode_data.(readdir("data/episode4", join = true))
data = vcat(data1, data2, data3, data4)

val_data = load_episode_data.(readdir("data/episode5", join = true))
test_data = load_episode_data.(readdir("data/episode6", join = true))

s, a = first(data)

dim = s.sol.total.dim
elements = size(dim)[1]
grid_size = maximum(dim.x)

z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))
fields = size(s.sol.total.u[1], 3)
h_fields = 64
z_fields = 2
activation = relu
design_size = 2 * length(vec(s.design))
h_size = 1024

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
design_encoder = DesignEncoder(design_size, h_size, z_elements, activation)
cell = WaveCell(nonlinear_latent_wave, runge_kutta)
dynamics = WaveDynamics(dim = OneDim(grid_size, z_elements); dynamics_kwargs...) |> gpu

mlp = Chain(
    Dense(3 * z_elements, h_size, activation), 
    Dense(h_size, h_size, activation),
    Dense(h_size, h_size, activation),
    Dense(h_size, 1)
    )

model = SigmaControlModel(wave_encoder, design_encoder, cell, dynamics, mlp) |> gpu

opt = Adam(0.0001)
ps = Flux.params(model)

train_loader = DataLoader(data, shuffle = true)

train_loss = Float32[]

path = mkpath("results/scattered_sigma/")

for i in 1:50
        
    epoch_loss = Float32[]

    for x in train_loader

        s, a = x[1]
        sol_scattered = gpu(s.sol.scattered)
        sigma_true = gpu(total_energy(sol_scattered)[2:end])

        Waves.reset!(model.z_dynamics)

        gs = Flux.gradient(ps) do
            loss = mse(sigma_true, model(gpu(s.sol.total), gpu(s.design), gpu(a)))
            Flux.ignore(() -> push!(epoch_loss, loss))
            return loss
        end

        Flux.Optimise.update!(opt, ps, gs)
    end

    push!(train_loss, mean(epoch_loss))
    println("Loss: $(train_loss[end])")

    plot_loss!(train_loss, path = joinpath(path, "train_loss.png"))
    # plot_predicted_sigma!(data, model, path = joinpath(path, "train_sigma.png"))
    
    plot_predicted_sigma!(val_data, model, path = joinpath(path, "val_sigma.png"))
    plot_predicted_sigma!(test_data, model, path = joinpath(path, "test_sigma.png"))
    BSON.@save joinpath(path, "model.bson")
end

