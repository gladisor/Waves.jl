using Waves, CairoMakie, Flux, BSON
using Optimisers
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(2)
display(Flux.device())

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)"

function make_plots(model::Waves.AcousticEnergyModel, batch; path::String, samples::Int = 1)
    s, a, t, y = batch
    y_hat = cpu(model(s, a, t))
    y = cpu(y)

    Waves.plot_latent_source(model, path = joinpath(path, "F.png"))

    for i in 1:min(length(s), samples)
        tspan = cpu(t[:, i])
        Waves.plot_predicted_energy(tspan, y[:, 1, i], y_hat[:, 1, i], title = "Total Energy", path = joinpath(path, "tot$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 2, i], y_hat[:, 2, i], title = "Incident Energy", path = joinpath(path, "inc$i.png"))
        Waves.plot_predicted_energy(tspan, y[:, 3, i], y_hat[:, 3, i], title = "Scattered Energy", path = joinpath(path, "sc$i.png"))
    end

    return nothing
end

function validate!(model::Waves.AcousticEnergyModel, val_loader::Flux.DataLoader)
    val_loss = []

    for batch in val_loader
        s, a, t, y = gpu(Flux.batch.(batch))
        loss = Flux.mse(model(s, a, t), y)
        push!(val_loss, loss)
    end

    return Flux.mean(val_loss)
end

function train!(model::Waves.AcousticEnergyModel, opt_state; train_loader::Flux.DataLoader, val_loader::Flux.DataLoader, val_every::Int, path::String = "")

    step = 0
    metrics = Dict(:train_loss => Vector{Float32}(), :val_loss => Vector{Float32}())
    train_loss_accumulator = Vector{Float32}()

    for epoch in 1:10
        for batch in train_loader
            s, a, t, y = gpu(Flux.batch.(batch))
            loss, back = Flux.pullback(m -> Flux.mse(m(s, a, t), y), model)
            @time gs = back(one(loss))[1]
            opt_state, model = Optimisers.update(opt_state, model, gs)
            push!(train_loss_accumulator, loss)
            step += 1

            if step % val_every == 0

                checkpoint_path = mkpath(joinpath(path, "checkpoint_step=$step"))
                s, a, t, y = gpu(Flux.batch.(first(val_loader)))
                make_plots(model, (s, a, t, y), path = checkpoint_path, samples = 4)

                val_loss = validate!(model, val_loader)
                push!(metrics[:train_loss], Flux.mean(train_loss_accumulator))
                push!(metrics[:val_loss], val_loss)
                empty!(train_loss_accumulator)
                println("Step: $(step), Train Loss: $(metrics[:train_loss][end]), Val Loss: $(metrics[:val_loss][end])")
                BSON.bson(joinpath(checkpoint_path, "checkpoint.bson"), model = cpu(model))
            end
        end
    end

    return model, opt_state
end

h_size = 256
nfreq = 500
elements = 1024
horizon = 1
batchsize = 32
latent_gs = 100.0f0
pml_width = 10.0f0
pml_scale = 10000.0f0
data_loader_kwargs = Dict(
    :batchsize => batchsize, 
    :shuffle => true, 
    :partial => false)

@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
@time data = [Episode(path = joinpath(DATA_PATH, "episode$i.bson")) for i in 1:10]
train_data, val_data = data[1:7], data[8:end]

train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

@time model = gpu(Waves.AcousticEnergyModel(;
    env, nfreq, h_size, pml_width, pml_scale, latent_dim = OneDim(latent_gs, elements)))
@time opt_state = Optimisers.setup(Optimisers.Adam(1f-4), model)

model, opt_state = train!(model, opt_state; 
    train_loader, val_loader, 
    val_every = 20, 
    path = "results")