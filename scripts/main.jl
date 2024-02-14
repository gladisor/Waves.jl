using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize

Flux.device!(0)
display(Flux.device())
Flux.CUDA.allowscalar(false)
println("Loaded Packages")

function energy_loss(model, s, a, t, y)
    return Flux.mse(model(s, a, t), y)
end

"""
measures average loss on dataset
"""
function validate!(model, val_loader::Flux.DataLoader, batches::Int; loss_func)
    val_loss = []

    for (i, batch) in enumerate(val_loader)
        s, a, t, y = gpu(Flux.batch.(batch))
        @time loss = loss_func(model, s, a, t, y)
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

function compute_gradients(model, loss_func, s, a, t, y)
    loss, back = Flux.pullback(m -> loss_func(m, s, a, t, y), model)
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
        )

    step = 0
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())
    train_loss_accumulator = Vector{Float32}()

    ## perform an initial gradient computation
    s, a, t, y = gpu(Flux.batch.(first(train_loader)))
    @time _, gs = compute_gradients(model, loss_func, s, a, t, y)
    gs_flat_accumulator, re = Flux.destructure(gs)
    gs_flat_accumulator .*= 0.0f0

    for epoch in 1:epochs
        for batch in train_loader
            s, a, t, y = gpu(Flux.batch.(batch))
            @time loss, gs = compute_gradients(model, loss_func, s, a, t, y)
            gs_flat, _ = Flux.destructure(gs)
            gs_flat_accumulator .+= gs_flat

            if step % accumulate == 0
                ∇ = re(gs_flat_accumulator ./ accumulate)
                opt_state, model = Optimisers.update(opt_state, model, ∇)
                gs_flat_accumulator .*= 0.0f0
            end

            push!(train_loss_accumulator, loss)
            step += 1

            if step % (val_every * accumulate) == 0

                ## creating checkpoint directory
                checkpoint_path = mkpath(joinpath(path, "checkpoint_step=$step"))

                ## save model checkpoint
                BSON.bson(joinpath(checkpoint_path, "checkpoint.bson"), model = cpu(model))

                ## plot some predictions
                make_plots(model, gpu(Flux.batch.(first(val_loader))), path = checkpoint_path, samples = val_samples)

                ## run validation
                @time val_loss = validate!(model, val_loader, val_batches, loss_func = loss_func)
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

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
## declaring hyperparameters
activation = leakyrelu
h_size = 256
in_channels = 4
nfreq = 500
elements = 1024
horizon = 1 #20
lr = 1f-4
batchsize = 4 #32 ## shorter horizons can use large batchsize
accumulate = 8
val_every = 20
val_batches = val_every
epochs = 10
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
latent_dim = OneDim(latent_gs, elements)
## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:500]
## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]
## preparing DataLoader(s)
train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")
## contstruct model & train
# @time model = gpu(AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim))
# @time model = gpu(NODEEnergyModel(env, activation, h_size, nfreq, latent_dim))
@time model = gpu(WaveControlPINN(;env, activation, h_size, nfreq, latent_dim))
loss_func = gpu(WaveControlPINNLoss(env, latent_dim))
@time opt_state = Optimisers.setup(Optimisers.Adam(lr), model)

# ## sample data
# s, a, t, y = gpu(Flux.batch.(first(train_loader)))

path = "models/wave_control_pinn_accumulate=$accumulate"
model, opt_state = @time train!(model, opt_state;
    accumulate = accumulate,
    train_loader,
    val_loader, 
    val_every,
    val_batches,
    epochs,
    path = joinpath(DATA_PATH, path),
    loss_func = loss_func
    )



