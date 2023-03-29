using Flux
Flux.CUDA.allowscalar(false)
using Flux: DataLoader
using Flux.Losses: mse, huber_loss, mae
using CairoMakie
using Statistics: mean
using ProgressMeter: @showprogress
using BSON

using Waves

include("models.jl")

function total_energy(sol::WaveSol)
    return sum.(energy.(displacement.(sol.u)))
end

function total_scattered_energy(s::WaveEnvState)
    return total_energy(s.sol.scattered)
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

function plot_loss!(train_loss::Vector{Float32}, test_loss::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, train_loss, label = "Train Loss")
    lines!(ax, test_loss, label = "Test Loss")

    axislegend(ax)
    save(path, fig)
end

data1 = load_episode_data.(readdir("data/episode1", join = true))
data2 = load_episode_data.(readdir("data/episode2", join = true))
data3 = load_episode_data.(readdir("data/episode3", join = true))
data4 = load_episode_data.(readdir("data/episode4", join = true))
data5 = load_episode_data.(readdir("data/episode5", join = true))
data = vcat(data1, data2, data3, data4, data5)

data6 = load_episode_data.(readdir("data/episode6", join = true))
data7 = load_episode_data.(readdir("data/episode7", join = true))
test_data = vcat(data6, data7)

val_episode1 = load_episode_data.(readdir("data/episode8", join = true))
val_episode2 = load_episode_data.(readdir("data/episode9", join = true))
val_episode3 = load_episode_data.(readdir("data/episode10", join = true))

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

s, a = first(data)

fields = 6
elements = 256
h_fields = 64
h_size = 1024
design_size = 2 * length(vec(s.design))
activation = tanh


model = SigmaControlModel(
    WaveEncoder(fields, h_fields, 2, activation),
    DesignEncoder(design_size, h_size, elements, activation),
    FEMIntegrator(elements, 100; grid_size = 4.0f0, dynamics_kwargs...),
    MLP(3 * elements, h_size, 2, 1, activation)) |> gpu

ps = Flux.params(model)
opt = Adam(0.0001)

path = mkpath("results/tanh/")
train_loader = DataLoader(data, shuffle = true)

train_loss = Float32[]
test_loss = Float32[]

for i in 1:50
    epoch_loss = Float32[]

    for x in train_loader
        s, a = gpu.(x[1])

        sigma_true = gpu(total_scattered_energy(s)[2:end])

        loss, gs = Flux.withgradient(ps) do
            return huber_loss(sigma_true, model(s, a))
        end

        Flux.Optimise.update!(opt, ps, gs)
        push!(epoch_loss, loss)
    end

    push!(train_loss, mean(epoch_loss))

    epoch_loss = Float32[]

    for x in test_data
        s, a = gpu.(x)
        sigma_true = gpu(total_scattered_energy(s)[2:end])
        sigma_pred = model(s, a)
        loss = huber_loss(sigma_true, sigma_pred)
        push!(epoch_loss, loss)
    end

    push!(test_loss, mean(epoch_loss))

    println("Train Loss: $(train_loss[end]), Test Loss: $(test_loss[end])")

    if i % 5 == 0
        plot_loss!(train_loss, test_loss, path = joinpath(path, "loss.png"))
        plot_predicted_sigma!(val_episode1, model, path = joinpath(path, "val_episode1.png"))
        plot_predicted_sigma!(val_episode2, model, path = joinpath(path, "val_episode2.png"))
        plot_predicted_sigma!(val_episode3, model, path = joinpath(path, "val_episode3.png"))
        BSON.@save joinpath(path, "model.bson") model
    end
end