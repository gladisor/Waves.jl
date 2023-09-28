using Waves, CairoMakie, Flux, BSON
using Optimisers
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

function make_plots(model::AcousticEnergyModel, batch; path::String, samples::Int = 1)
    s, a, t, y = batch
    y_hat = cpu(model(s, a, t))
    y = cpu(y)
    # Waves.plot_latent_source(model, path = joinpath(path, "F.png"))
    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        Waves.plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    return nothing
end

"""
measures average loss on dataset
"""
function validate!(model::AcousticEnergyModel, val_loader::Flux.DataLoader)
    val_loss = []

    for batch in val_loader
        s, a, t, y = gpu(Flux.batch.(batch))
        loss = Flux.mse(model(s, a, t), y)
        push!(val_loss, loss)
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

function train!(model::AcousticEnergyModel, opt_state; train_loader::Flux.DataLoader, val_loader::Flux.DataLoader, val_every::Int, epochs::Int, path::String = "")

    step = 0
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())
    train_loss_accumulator = Vector{Float32}()

    for epoch in 1:epochs
        for batch in train_loader
            s, a, t, y = gpu(Flux.batch.(batch))
            loss, back = Flux.pullback(m -> Flux.mse(m(s, a, t), y), model)
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
                val_loss = validate!(model, val_loader)
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

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)"
## declaring hyperparameters
h_size = 256
nfreq = 500
elements = 1024

horizon = 10
horizon = 1

batchsize = 64
val_every = 20
epochs = 10

latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0

train_val_split = 0.9 ## choosing percentage of data for val
data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

## loading environment and data
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
# @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:500]
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 10:20]

## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

## contstruct model & train
@time model = gpu(AcousticEnergyModel(;env, nfreq, h_size, pml_width, pml_scale, latent_dim = OneDim(latent_gs, elements)))
# model = gpu(BSON.load("tranable_source/checkpoint_step=2640/checkpoint.bson")[:model])
@time opt_state = Optimisers.setup(Optimisers.Adam(1f-4), model)

path = "h_size=$(h_size)_latent_gs=$(latent_gs)_pml_width=$(pml_width)_nfreq=$nfreq"
model, opt_state = @time train!(model, opt_state;
    train_loader, 
    val_loader, 
    val_every,
    epochs,
    path = path
    # path = joinpath(DATA_PATH, path)
    )



# s, a, t, y = gpu(Flux.batch.(first(train_loader)))
# @time z = cpu(Waves.generate_latent_solution(model, s, a, t))
# @time y_hat = cpu(model(s, a, t))

# latent_dim = cpu(model.iter.dynamics.dim)
# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, cpu(t[:, 1]), cpu(y_hat[:, 1, 1]))
# lines!(ax, cpu(t[:, 1]), cpu(y_hat[:, 2, 1]))
# lines!(ax, cpu(t[:, 1]), cpu(y_hat[:, 3, 1]))
# save("energy.png", fig)

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -3.0f0, 3.0f0)
# record(fig, "latent.mp4", axes(t, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
# end

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, -100.0f0, -50.0f0)
# lines!(ax, latent_dim.x, cpu(model.iter.dynamics.pml))
# save("pml.png", fig)