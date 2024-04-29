using CairoMakie, BSON
using Images: imresize
using DataFrames
using CSV
using Distributed

desired_worker_count = 4
current_worker_count = nprocs()

if current_worker_count < desired_worker_count
    additional_worker_count = desired_worker_count - current_worker_count
    addprocs(additional_worker_count; exeflags = "--project=$(Base.active_project())")
elseif current_worker_count > desired_worker_count
    excess_worker_count = current_worker_count - desired_worker_count
    rmprocs(workers()[1:excess_worker_count])
end

@everywhere using Distributed, Flux, Waves, Optimisers, BSON, CUDA
Flux.CUDA.allowscalar(false)
println("Loaded Packages, workers added.")

@everywhere function energy_loss(model, s, a, t, y)
    return Flux.mse(model(s, a, t), y)

    # mse = CUDA.cpu(Flux.mse(model(s, a, t), y))
    # return gpu(mse)

    # CUDA.@allowscalar begin
    #     # The operation causing the issue, just for debugging purposes
    #     # x = Flux.mse(model(s, a, t), y)
    #     print("222")
    #     aa = model(s, a, t)
    #     print("2222")
        
    #     x = Flux.mse(aa, y)
    # end
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

@everywhere function compute_gradients(model, loss_func, s, a, t, y)
    loss, back = Flux.pullback(m -> loss_func(m, s, a, t, y), model)
    println("22")
    gs = back(one(loss))[1]
    return loss, gs
end

function train!(;
        loss_func,
        accumulate::Int,
        train_loader,
        val_loader::Flux.DataLoader, 
        val_every::Int, 
        val_batches::Int, 
        val_samples::Int = 4,
        epochs::Int,
        path::String = "",
        )
    
    @everywhere m = initialize_model()
    @everywhere models = [deepcopy(m) for i in 1:number_of_gpus]
    @everywhere opt_state = initialize_optimiser(m)

    step = 0
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())
    CSV.write(joinpath(mkpath(path), "loss_data.csv"), DataFrame(["step" "train loss" "val loss"], :auto))
    train_loss_accumulator = Vector{Float32}()

    @everywhere begin
        # Initialize a persistent dictionary to track state per worker if needed
        const state = Dict{String, Any}()
        state["first_run"] = true
    end
    
    for epoch in 1:epochs
        for batches in zip(train_loader...)

            results_task = @sync @distributed for (i, w) in collect(enumerate(workers()))
                try
                    gpu_id = w % 4 
                    Flux.device!(gpu_id)
                    model = models[gpu_id + 1]
                    if state["first_run"]
                        s, a, t, y = gpu(Flux.batch.(batches[gpu_id]))
                        println("2")
                        @time _, gs = compute_gradients(model, loss_func, s, a, t, y)
                        println("3")
                        gs_flat_accumulator, re = Flux.destructure(gs)
                        println("4")
                        gs_flat_accumulator .*= 0.0f0
                        println("5")
                        state["first_run"] = false
                    end

                    s, a, t, y = gpu(Flux.batch.(batches[gpu_id]))
                    @time loss, gs = compute_gradients(model, loss_func, s, a, t, y)
                    return Dict(:error => e, :worker => w)
                catch e
                    Dict(:error => e, :worker => w)
                end
            end
            results = fetch(results_task)
            if any(key == :error for key in keys(results))
                return results
            end

            for result in results
                loss, gs = result
                gs_flat, _ = Flux.destructure(gs)
                gs_flat_accumulator .+= gs_flat

                if step % accumulate == 0
                    ∇ = re(gs_flat_accumulator ./ accumulate)
                    opt_state, m = Optimisers.update(opt_state, m, ∇)
                    gs_flat_accumulator .*= 0.0f0
                end
    
                push!(train_loss_accumulator, loss)
                step += 1
            end

            @everywhere models = [deepcopy(m) for i in 1:number_of_gpus]

            if (step - 1) % (val_every * accumulate) == 0

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

                ## save step to CSV file
                step_data = [step metrics[:train_loss][end] metrics[:val_loss][end]]
                CSV.write(joinpath(path, "loss_data.csv"), DataFrame(step_data, :auto), append=true)
            end
        end
    end

    return model, opt_state
end

@everywhere function initialize_model()
    env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
    h_size = 256
    in_channels = 4
    nfreq = 500
    pml_width = 10.0f0
    pml_scale = 10000.0f0
    elements = 1024
    latent_gs = 100.0f0
    latent_dim = OneDim(latent_gs, elements)
    return AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim)
end

@everywhere function initialize_optimiser(model)
    lr = 1f-4
    return Optimisers.setup(Optimisers.Adam(lr), model)
end

@everywhere dataset_name = "dataset_200"
@everywhere DATA_PATH = "scratch/$dataset_name"
## declaring hyperparameters
activation = leakyrelu
lr = 1f-4
horizon = 20
batchsize = 8 #32 ## shorter horizons can use large batchsize
accumulate = 1
val_every = 20
val_batches = val_every
epochs = 10
train_val_split = 0.90 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)
## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:10]
## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]
@everywhere number_of_gpus = nprocs()
split_indices = vcat(collect(1:div(length(train_data), number_of_gpus):length(train_data)), length(train_data)+1)
train_data_gpu_split = [train_data[split_indices[i]:split_indices[i+1]-1] for i in 1:(length(split_indices)-1)]
## preparing DataLoader(s)
train_loader = [Flux.DataLoader(prepare_data(train_data_gpu_split[i], horizon); data_loader_kwargs...) for i in 1:length(train_data_gpu_split)]
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)*length(train_loader[1])), Val Batches: $(length(val_loader))")
## train
path = "models/acoustic_energy_ViT_horizon=$horizon,lr=$lr"
ret = @time train!(;
# model, opt_state = @time train!(;
    accumulate = accumulate,
    train_loader,
    val_loader, 
    val_every,
    val_batches,
    epochs,
    path = joinpath(DATA_PATH, path),
    loss_func = energy_loss
    )


