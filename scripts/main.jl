using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize

Flux.device!(1)
display(Flux.device())
Flux.CUDA.allowscalar(false)

println("Loaded Packages")
include("random_pos_gaussian_source.jl")

function render_latent_solution!(dim::OneDim, t::Vector{Float32}, z::Array{Float32, 3}; path::String)
    tot = z[:, 1, :]
    inc = z[:, 3, :]
    sc = tot .- inc

    fig = Figure()
    ax = Axis(fig[1, 1])
    xlims!(ax, dim.x[1], dim.x[end])
    ylims!(ax, -2.0f0, 2.0f0)
    
    record(fig, joinpath(path, "sc.mp4"), axes(t, 1)) do i
        empty!(ax)
        lines!(ax, dim.x, sc[:, i], color = :blue)
    end
end

function make_plots(
        model::AcousticEnergyModel, 
        batch; path::String, 
        samples::Int = 1)

    s, a, t, y = batch
    z = cpu(Waves.generate_latent_solution(model, s, a, t))
    latent_dim = cpu(model.iter.dynamics.dim)
    render_latent_solution!(latent_dim, cpu(t[:, 1]), z[:, :, 1, :], path = path)

    z0, (C, F, PML) = Waves.get_parameters_and_initial_condition(model, s, a, t)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, latent_dim.x, cpu(PML[:, 1]))
    save(joinpath(path, "pml.png"), fig)

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, latent_dim.x, cpu(F.shape[:, 1]))
    save(joinpath(path, "force.png"), fig)

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

function train!(
        model,
        opt_state; 
        train_loader::Flux.DataLoader, 
        val_loader::Flux.DataLoader, 
        val_every::Int, 
        val_batches::Int, 
        epochs::Int,
        path::String = "",
        loss_func
        )

    step = 0
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())
    train_loss_accumulator = Vector{Float32}()

    for epoch in 1:epochs
        for batch in train_loader
            s, a, t, y = gpu(Flux.batch.(batch))
            loss, back = Flux.pullback(m -> loss_func(m, s, a, t, y), model)
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
                make_plots(model, 
                    gpu(Flux.batch.(first(val_loader))), 
                    path = checkpoint_path, samples = 4)

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
horizon = 20
lr = 1f-4
batchsize = 32 ## shorter horizons can use large batchsize
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
include("node.jl")
include("wave_control_pinn.jl")

@time model = gpu(AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim))

# @time model = gpu(NODEEnergyModel(env, activation, h_size, nfreq, latent_dim))
# base = build_cnn_base(env, in_channels, activation, h_size)
# head = build_pinn_wave_encoder_head(h_size, activation, nfreq, latent_dim)
# W = WaveEncoder(base, head)
# D = DesignEncoder(env, h_size, activation, nfreq, latent_dim)
# compressor_size = 16
# R = build_compressor(8, h_size, activation, compressor_size)
# U = Chain(
#     Dense(compressor_size + 2, h_size, activation),
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation), 
#     Dense(h_size, h_size, activation),
#     Dense(h_size, 4))

# tspan = build_tspan(0.0f0, env.dt, env.integration_steps)
# pinn_grid = build_pinn_grid(latent_dim, tspan)

# model = gpu(WaveControlPINN(
#     W, D, R, U, 
#     pinn_grid,
#     env.source.freq,
#     get_dx(latent_dim)))

# loss_func = gpu(WaveControlPINNLoss(
#     env.iter.dynamics.c0, 
#     env.source.freq,
#     Matrix{Float32}(Waves.gradient(latent_dim.x)),
#     Matrix{Float32}(Waves.gradient(tspan)),
#     build_dirichlet(latent_dim))
#     )

## sample data
# s, a, t, y = gpu(Flux.batch.(first(train_loader)))

# MODEL_PATH = "/scratch/cmpe299-fa22/tristan/data/variable_source_yaxis_x=-10.0/models/horizon=20,lr=0.0001/checkpoint_step=6120/checkpoint.bson"
# model = gpu(BSON.load(MODEL_PATH)[:model])
@time opt_state = Optimisers.setup(Optimisers.Adam(lr), model)
path = "models/ours_balanced_field_scale"
model, opt_state = @time train!(model, opt_state;
    train_loader,
    val_loader, 
    val_every,
    val_batches,
    epochs,
    path = joinpath(DATA_PATH, path),
    loss_func = energy_loss
    )