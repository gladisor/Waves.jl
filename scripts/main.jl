using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(3)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")

function make_plots(model::AcousticEnergyModel, loss_func, s, a, t, y; path::String, samples::Int)
    render_latent_solution(model, s, a, t; path)

    y_hat = cpu(model(s, a, t))
    y = cpu(y)
    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        Waves.plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    return nothing
end

function make_plots(
        model::WaveReconstructionModel, 
        loss_func::WaveReconstructionLoss, 
        s, a, t, y, w; 
        path::String, 
        samples::Int)

    make_plots(model.energy_model, loss_func, s, a, t, y, path = path, samples = samples)
    image_path = mkpath(joinpath(path, "images"))
    _, w_hat = cpu(model(s, a, t, loss_func.idx))

    w_hat = cpu(w_hat)
    w = cpu(w)
    tspan = cpu(t[loss_func.idx, :])

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Predicted")
    ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Ground Truth")

    for i in axes(w, 3)
        empty!(ax1)
        empty!(ax2)
        heatmap!(ax1, w_hat[:, :, i, 1], colormap = :ice)
        heatmap!(ax2, w[:, :, i, 1], colormap = :ice)
        save(joinpath(image_path, "$i.png"), fig)
    end
end

function make_plots(model::WaveReconstructionModel, loss_func::LatentConsistencyLoss, args...; kwargs...)
    return make_plots(model, loss_func.reconstruction_loss, args...; kwargs...)
end

"""
measures average loss on dataset
"""
function validate(model, loss_func, val_loader::Flux.DataLoader, batches::Int)
    val_loss = []

    for (i,batch) in enumerate(val_loader)
        batch = gpu(batch)
        @time loss = loss_func(model, batch...)
        push!(val_loss, loss)
        println("Val Batch: $i")

        if i == batches
            break
        end
    end

    return Flux.mean(val_loss)
end

function plot_loss(metrics::Dict, val_every::Int; path::String)

    steps = collect(1:length(metrics[:train_loss])) * val_every

    fig = Figure(resolution = (1920, 1080))
    ax1 = Axis(fig[1, 1], xlabel = "Batch Update", ylabel = "Average Loss")
    lines!(ax1, steps, metrics[:train_loss], color = :blue, label = "Train")
    lines!(ax1, steps, metrics[:val_loss], color = :orange, label = "Val")
    axislegend(ax1)

    ax2 = Axis(fig[1, 2], xlabel = "Batch Update", ylabel = "Log Scale Average Loss")
    lines!(ax2, steps, log.(metrics[:train_loss]), color = :blue, label = "Train")
    lines!(ax2, steps, log.(metrics[:val_loss]), color = :orange, label = "Val")
    axislegend(ax2)
    save(path, fig)
end

function train!(;
        model, opt_state, loss_func,
        train_loader::Flux.DataLoader, val_loader::Flux.DataLoader, 
        val_every::Int, val_batches::Int, 
        epochs::Int, path::String)

    step = 0
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())
    train_loss_accumulator = Vector{Float32}()

    for epoch in 1:epochs
        for batch in train_loader
            batch = gpu(batch)
            loss, back = Flux.pullback(m -> loss_func(m, batch...), model)
            @time gs = back(one(loss))[1]
            opt_state, model = Optimisers.update(opt_state, model, gs)
            push!(train_loss_accumulator, loss)
            step += 1

            if step % val_every == 0

                ## creating checkpoint directory
                checkpoint_path = mkpath(joinpath(path, "checkpoint_step=$step"))

                ## save model checkpoint
                BSON.bson(joinpath(checkpoint_path, "checkpoint.bson"), model = cpu(model))

                ## plot some predictions
                make_plots(
                    model,
                    loss_func,
                    gpu(first(val_loader))...;
                    path = checkpoint_path,
                    samples = 4)

                ## run validation
                @time val_loss = validate(model, loss_func, val_loader, val_batches)
                push!(metrics[:train_loss], Flux.mean(train_loss_accumulator))
                push!(metrics[:val_loss], val_loss)
                empty!(train_loss_accumulator)

                ## plot the losses
                plot_loss(metrics, val_every, path = joinpath(checkpoint_path, "loss.png"))

                ## print to command line
                println("Step: $(step), Train Loss: $(metrics[:train_loss][end]), Val Loss: $(metrics[:val_loss][end])")
            end
        end
    end

    return model, opt_state
end

function prepare_reconstruction_data(ep::Episode{S, Matrix{Float32}}, horizon::Int) where S
    s = S[]
    a = Vector{<: AbstractDesign}[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]
    w = Array{Float32, 3}[]
    
    n = horizon - 1
    for i in 1:(length(ep) - n - 1)
        boundary = i + n
        push!(s, ep.s[i])
        push!(a, ep.a[i:boundary])

        tspan = flatten_repeated_last_dim(hcat(ep.t[i:boundary]...))
        push!(t, tspan)

        signal = cat(ep.y[i:boundary]..., dims = 3)
        signal = permutedims(flatten_repeated_last_dim(permutedims(signal, (2, 1, 3))))
        push!(y, signal)

        push!(w, cat([ep.s[j].wave for j in i+1:boundary+1]..., dims = 3))
    end

    return s, a, t, y, w
end

function prepare_reconstruction_data(args...)
    return vcat.(prepare_reconstruction_data.(args...)...)
end

function get_reconstruction_data(ep::Episode, horizon::Int, idx::Int)
    boundary = idx + horizon - 1
    s = ep.s[idx]
    a = ep.a[idx:boundary]
    t = flatten_repeated_last_dim(hcat(ep.t[idx:boundary]...))
    y = cat(ep.y[idx:boundary]..., dims = 3)
    y = permutedims(flatten_repeated_last_dim(permutedims(y, (2, 1, 3))))
    w = cat([ep.s[j].wave for j in idx+1:boundary+1]..., dims = 3)
    return (s, a, t, y, w)
end

struct EpisodeDataset
    episodes::Vector{Episode}
    horizon::Int
end

function Base.length(dataset::EpisodeDataset)
    return sum(length.(dataset.episodes) .- dataset.horizon)
end

function Base.getindex(dataset::EpisodeDataset, idx::Int)
    episode_length = length(dataset.episodes[1]) - dataset.horizon
    episode_idx = (idx - 1) ÷ episode_length + 1
    data_idx = (idx - 1) % episode_length + 1
    return get_reconstruction_data(dataset.episodes[episode_idx], horizon, data_idx)
end

function Base.getindex(dataset::EpisodeDataset, idxs::AbstractVector)
    Flux.batch.(zip([getindex(dataset, idx) for idx in idxs]...))
end

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)"
## declaring hyperparameters
h_size = 256
activation = leakyrelu
nfreq = 500
elements = 1024
horizon = 20
batchsize = 4 #32
val_every = 20
val_batches = val_every
epochs = 10
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
# @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:500]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:20]

## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
train_loader = Flux.DataLoader(EpisodeDataset(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(EpisodeDataset(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

## contstruct model & train
latent_dim = OneDim(latent_gs, elements)
wave_encoder = WaveEncoder(env, h_size, activation, nfreq, latent_dim)
design_encoder = DesignEncoder(env, h_size, activation, nfreq, latent_dim)
dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, pml_width, pml_scale)
iter = Integrator(runge_kutta, dyn, env.dt)
energy_model = AcousticEnergyModel(wave_encoder, design_encoder, iter, get_dx(latent_dim), env.source.freq)

# ## grab sample data
# model = gpu(WaveReconstructionModel(energy_model, activation))
# # loss_func = gpu(WaveReconstructionLoss(env, horizon))
# loss_func = gpu(LatentConsistencyLoss(env, horizon))
# opt_state = Optimisers.setup(Optimisers.Adam(1f-4), model)
# # s, a, t, y, w = gpu(first(val_loader))

# path = "latent_consistency_loss_horizon=$(horizon)_batchsize=$(batchsize)_h_size=$(h_size)_latent_gs=$(latent_gs)_pml_width=$(pml_width)_nfreq=$nfreq"
# model, opt_state = @time train!(;
#     model, 
#     opt_state,
#     loss_func,
#     train_loader, 
#     val_loader, 
#     val_every,
#     val_batches,
#     epochs,
#     path = joinpath(DATA_PATH, path)
#     # path = "testing"
#     )