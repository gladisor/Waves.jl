using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using DataFrames
using CSV
using ChainRulesCore: ignore_derivatives

# include("../src/model/latent_transformation_model.jl")

Flux.CUDA.allowscalar(false)
println("Loaded Packages")

const_alpha = 1.0f0
var_alpha_consistency = 1 * ones(40000)
var_alpha_energy = const_alpha * ones(40000)

function energy_loss(model::AcousticEnergyModel, z, y, energy_indices)
    y_hat = compute_latent_energy(z, model.dx)
    return Flux.mse(y_hat, y[:,energy_indices,:]) 
end

function energy_loss(model::NODEEnergyModel, z, y, energy_indices)
    # energy = sum(z .^ 2, dims = 1) * model.dx
    # y_hat = permutedims(energy[1, :, :, :], (3, 1, 2))
    y_hat = permutedims(z[:, 1, :, :], (3, 1, 2))
    return Flux.mse(y_hat, y[:,energy_indices,:]) 
end

function energy_loss(model, s, a, t, y, s_, energy_indices)
    z = generate_latent_solution(model, s, a, t)
    return energy_loss(model, z, y, energy_indices)
end

function compute_latent_states(model, s_)
    return cat([model.wave_encoder(s_[i, :])[:, 1:4, :] for i in 1:size(s_, 1)]..., dims=4)
end

function consistency_loss(model, z, s_)
    # latent_states = ignore_derivatives(compute_latent_states(model, s_))
    # return Flux.mse(z[:,:,:,101:100:end], latent_states)
    return 0
end

function consistency_loss(model, s, a, t, y, s_, energy_indices)
    # z = generate_latent_solution(model, s, a, t)
    # return consistency_loss(model, z, s_)
    return 0
end

function total_loss(model, s, a, t, y, s_, step::Int, energy_indices)
    z = generate_latent_solution(model, s, a, t)
    energy_component = energy_loss(model, z, y, energy_indices)
    consistency_component = consistency_loss(model, z, s_)
    return energy_component * var_alpha_energy[step] + consistency_component * var_alpha_consistency[step]
end

function validate!(model, val_loader::Flux.DataLoader, batches::Int, energy_indices; loss_func::Function)
    val_loss = []

    for (i, batch) in enumerate(val_loader)
        s, a, t, y, s_ = gpu(Flux.batch.(batch))
        @time loss = loss_func(model, s, a, t, y, s_, energy_indices)
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

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Batch Update", ylabel = "Average Loss")
    lines!(ax, steps, metrics[:train_loss], color = :blue, label = "Train")
    lines!(ax, steps, metrics[:val_loss], color = :orange, label = "Val")
    axislegend(ax)
    save(path, fig)
end

function compute_gradients(model, loss_func, s, a, t, y, s_, step::Int, energy_indices)
    loss, back = Flux.pullback(m -> loss_func(m, s, a, t, y, s_, step, energy_indices), model)
    gs = back(one(loss))[1]
    return loss, gs
end

function train!(
        model,
        opt_state; 
        loss_func,
        accumulate::Int,
        train_loader::Flux.DataLoader,
        val_loader::Flux.DataLoader, 
        val_every::Int, 
        val_batches::Int, 
        val_samples::Int = 4,
        epochs::Int,
        path::String = "",
        energy_indices
        )

    step = 1
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}(), :energy => Vector{Float32}(), :consistency => Vector{Float32}(), :val_energy => Vector{Float32}(), :val_consistency => Vector{Float32}())
    CSV.write(joinpath(mkpath(path), "loss_data.csv"), DataFrame(["step" "train loss" "val loss" "energy loss" "val energy loss" "consistency loss" "val consistency loss"], :auto))
    train_loss_accumulator = Vector{Float32}()
    energy_loss_accumulator = Vector{Float32}()
    consistency_loss_accumulator = Vector{Float32}()

    ## perform an initial gradient computation
    s, a, t, y, s_ = gpu(Flux.batch.(first(train_loader)))
    @time _, gs = compute_gradients(model, loss_func, s, a, t, y, s_, step, energy_indices)
    gs_flat_accumulator, re = Flux.destructure(gs)
    gs_flat_accumulator .*= 0.0f0

    for epoch in 1:epochs
        for batch in train_loader
            s, a, t, y, s_ = gpu(Flux.batch.(batch))
            @time loss, gs = compute_gradients(model, loss_func, s, a, t, y, s_, step, energy_indices)
            gs_flat, _ = Flux.destructure(gs)
            gs_flat_accumulator .+= gs_flat

            push!(train_loss_accumulator, loss)
            z = generate_latent_solution(model, s, a, t)
            push!(energy_loss_accumulator, energy_loss(model, z, y, energy_indices)*var_alpha_energy[step])
            push!(consistency_loss_accumulator, consistency_loss(model, z, s_)*var_alpha_consistency[step])

            if step % accumulate == 0
                ∇ = re(gs_flat_accumulator ./ accumulate)
                opt_state, model = Optimisers.update(opt_state, model, ∇)
                gs_flat_accumulator .*= 0.0f0
            end

            step += 1

            if step % (val_every * accumulate) == 0

                ## creating checkpoint directory
                checkpoint_path = mkpath(joinpath(path, "checkpoint_step=$step"))

                ## save model checkpoint
                BSON.bson(joinpath(checkpoint_path, "checkpoint.bson"), model=cpu(model))

                ## plot some predictions
                Waves.make_plots(model, gpu(Flux.batch.(first(val_loader))), path = checkpoint_path, samples = val_samples)

                ## run validation
                @time val_energy_loss = validate!(model, val_loader, val_batches, energy_indices; loss_func=energy_loss)
                @time val_consistency_loss = validate!(model, val_loader, val_batches, energy_indices; loss_func=consistency_loss)
                val_energy_loss *= var_alpha_energy[step]
                val_consistency_loss *= var_alpha_consistency[step]

                push!(metrics[:train_loss], Flux.mean(train_loss_accumulator))
                push!(metrics[:val_loss], val_energy_loss + val_consistency_loss)
                push!(metrics[:val_energy], val_energy_loss)
                push!(metrics[:val_consistency], val_consistency_loss)
                push!(metrics[:energy], Flux.mean(energy_loss_accumulator))
                push!(metrics[:consistency], Flux.mean(consistency_loss_accumulator))
                empty!(train_loss_accumulator)
                empty!(consistency_loss_accumulator)
                empty!(energy_loss_accumulator)

                ## plot the losses
                plot_loss(metrics, val_every, path = joinpath(checkpoint_path, "loss.png"))

                ## print to command line
                println("Step: $(step), Train Loss: $(metrics[:train_loss][end]), Val Loss: $(metrics[:val_loss][end])")
                println("Energy Loss: $(metrics[:energy][end]), Consistency Loss: $(metrics[:consistency][end])")
                println("Val Energy Loss: $(metrics[:val_energy][end]), Val Consistency Loss: $(metrics[:val_consistency][end])")

                ## save step to CSV file
                step_data = [step metrics[:train_loss][end] metrics[:val_loss][end] metrics[:energy][end] metrics[:val_energy][end] metrics[:consistency][end] metrics[:val_consistency][end]]
                CSV.write(joinpath(path, "loss_data.csv"), DataFrame(step_data, :auto), append=true)
                cp(joinpath(path, "loss_data.csv"), joinpath(checkpoint_path, "loss_data.csv"), force=true)
            end
        end
        println("Epoch $epoch done!")
    end

    return model, opt_state
end

struct Hyperparameters
    dataset_name::String
    activation::Function
    h_size::Int
    in_channels::Int
    nfreq::Int
    elements::Int
    horizon::Int
    lr::Float32
    batchsize::Int
    accumulate::Int
    val_every::Int
    val_batches::Int
    epochs::Int
    latent_gs::Float32
    pml_width::Float32
    pml_scale::Float32
    train_val_split::Float32
    regressed_signals::Vector{Int64}
end

function log_hyperparameters(params::Hyperparameters)
    println("~Hyperparameters~")
    println("Dataset Name: $(params.dataset_name)")
    println("Activation: $(params.activation)")
    println("Hidden Size: $(params.h_size)")
    println("In Channels: $(params.in_channels)")
    println("NFreq: $(params.nfreq)")
    println("Elements: $(params.elements)")
    println("Horizon: $(params.horizon)")
    println("Learning Rate: $(params.lr)")
    println("Batch Size: $(params.batchsize)")
    println("Accumulate: $(params.accumulate)")
    println("Validation Every: $(params.val_every)")
    println("Validation Batches: $(params.val_batches)")
    println("Epochs: $(params.epochs)")
    println("Latent GS: $(params.latent_gs)")
    println("PML Width: $(params.pml_width)")
    println("PML Scale: $(params.pml_scale)")
    println("Train/Val Split: $(params.train_val_split)")
    println(""" Used variable alpha:
        var_alpha_consistency = 1 * ones(40000) * 0
        var_alpha_energy = $const_alpha * ones(40000)
    """)
    println("Regressed Signals Indices: $(params.regressed_signals)")
    println("~~~")
end

Flux.device!(0)
display(Flux.device())
# dataset_name = "pos_adjustment_masked_M=1"
# dataset_name = "dataset_pos_adjustment_masked"
# dataset_name = "fully_adjustable_masked_M=2"
# dataset_name = "full_adjustment_masked_signals_M=2"
# dataset_name = "pos_adjustment_masked_signals_M=4"
# dataset_name = "pos_adjustment_masked_signals_M=2"
dataset_name = "pos_adjustment_masked_signals_M=1"
DATA_PATH = "scratch/$dataset_name"
## declaring hyperparameters
activation = leakyrelu
h_size = 256
in_channels = 4
nfreq = 500
elements = 1024 # "default" = 1024
horizon = 10
lr = 1f-5
batchsize = horizon >= 5 ? 64 : 256
# batchsize = 256 #32 ## shorter horizons can use large batchsize
accumulate = 1
val_every = 100
val_batches = val_every
epochs = horizon >= 5 ? 20 : 35
# epochs = 20
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
latent_dim = OneDim(latent_gs, elements)
focusing = false
# 1, 2, 3 - total, incident, scattered (in entire grid)
# 4:7 - total energy signal in each of the quadrants
# 8:11 - incident energy signal in each of the quadrants
# 12:15 - scattered energy signal in each of the quadrants
regressed_signals = focusing ? [5, 9, 13] : [1, 2, 3] # total, incident, scattered (in upper right quadrant)
## logging hyperparameters
hp = Hyperparameters(dataset_name, activation, h_size, in_channels, nfreq, elements, horizon, lr, batchsize, accumulate, val_every, val_batches, epochs, latent_gs, pml_width, pml_scale, train_val_split, regressed_signals)
log_hyperparameters(hp)
## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:500]
# @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 501:1000]
## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]
## preparing DataLoader(s)
train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")
## contstruct model & train
# @time model = gpu(AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim, base_function=build_cnn_base))
@time model = gpu(NODEEnergyModel(env, activation, h_size, nfreq, latent_dim))
# MODEL_PATH = "scratch/pos_adjustment_masked_M=2/models/AEM_batchsize=32_jobID=42391/checkpoint_step=12000/checkpoint.bson"
# model = gpu(BSON.load(MODEL_PATH)[:model])
@time opt_state = Optimisers.setup(Optimisers.Adam(lr), model)
job_id = length(ARGS) == 0 ? 0 : ARGS[1]
model_type = typeof(model) == NODEEnergyModel ? "NODE" : "AEM"
path = "models/$(model_type)_$(focusing ? "focusing" : "suppression")_jobID=$(job_id)"
model, opt_state = @time train!(model, opt_state;
    accumulate = accumulate,
    train_loader,
    val_loader, 
    val_every,
    val_batches,
    epochs,
    path = joinpath(DATA_PATH, path),
    loss_func = total_loss,
    energy_indices = regressed_signals
    )