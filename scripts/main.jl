using JLD2
using Flux
Flux.CUDA.allowscalar(false)
using Flux: DataLoader
using Flux.Losses: mse, huber_loss, mae
using CairoMakie
using Statistics: mean
using ProgressMeter: @showprogress

using Waves

include("design_encoder.jl")

function load_episode_data(path::String)
    file = jldopen(path)
    s = file["s"]
    a = file["a"]
    return (s, a)
end

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

function plot_predicted_sigma!(data::Vector{Tuple{WaveEnvState{Scatterers}, Scatterers}}, model::SigmaControlModel; path::String)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Energy: σ", title = "Scattered Energy of Solution over Time")

    s, a = data[1]

    sigma_true = total_energy(s.sol.scattered)[2:end]
    sigma_pred = model(s.sol.total, s.design, a)
    lines!(ax, cpu(s.sol.total.t[2:end]), cpu(sigma_true), color = :blue, label = "True Energy")
    lines!(ax, cpu(s.sol.total.t[2:end]), cpu(sigma_pred), color = :orange, label = "Model Predicted Energy")

    for d in data[2:end]
        s, a = d
        sigma_true = total_energy(s.sol.scattered)[2:end]
        sigma_pred = model(s.sol.total, s.design, a)
        lines!(ax, cpu(s.sol.total.t[2:end]), cpu(sigma_true), color = :blue)
        lines!(ax, cpu(s.sol.total.t[2:end]), cpu(sigma_pred), color = :orange)
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

data = load_episode_data.(readdir("data/episode1", join = true))
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
h_size = 128

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
design_encoder = DesignEncoder(design_size, h_size, z_elements, activation)
cell = WaveCell(nonlinear_latent_wave, runge_kutta)
dynamics = WaveDynamics(dim = OneDim(grid_size, z_elements); dynamics_kwargs...) |> gpu

mlp = Chain(
    Dense(3 * z_elements, h_size, activation), 
    Dense(h_size, h_size, activation),
    Dense(h_size, 1))

model = SigmaControlModel(wave_encoder, design_encoder, cell, dynamics, mlp) |> gpu

opt = Adam(0.0001)
ps = Flux.params(model)

train_loader = DataLoader(data, shuffle = true)

train_loss = Float32[]

for i in 1:50
        
    epoch_loss = Float32[]

    @showprogress for x in train_loader

        s, a = x[1]
        sol_total = gpu(s.sol.total)
        sol_scattered = gpu(s.sol.scattered)
        sigma_true = total_energy(sol_scattered)[2:end]

        design = gpu(s.design)
        a = gpu(a)

        Waves.reset!(model.z_dynamics)

        gs = Flux.gradient(ps) do
            loss = mae(sigma_true, model(sol_total, design, a))
            Flux.ignore(() -> push!(epoch_loss, loss))
            return loss
        end

        Flux.Optimise.update!(opt, ps, gs)
    end

    push!(train_loss, mean(epoch_loss))
    println("Loss: $(train_loss[end])")

    plot_loss!(train_loss, path = "train_loss.png")
    plot_predicted_sigma!(data, model, path = "train_sigma.png")
end

