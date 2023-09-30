using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

function make_plots(model::AcousticEnergyModel, batch; path::String, samples::Int = 1)
    s, a, t, y = batch

    z = cpu(Waves.generate_latent_solution(model, s, a, t))
    latent_dim = cpu(model.iter.dynamics.dim)

    fig = Figure()
    ax = Axis(fig[1, 1])
    xlims!(ax, latent_dim.x[1], latent_dim.x[end])
    ylims!(ax, -2.0f0, 2.0f0)

    record(fig, joinpath(path, "latent.mp4"), axes(t, 1)) do i
        empty!(ax)
        lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
    end

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

struct LocalizationLayer
    coords::AbstractArray{Float32, 4}
end

Flux.@functor LocalizationLayer
Flux.trainable(::LocalizationLayer) = (;)

function LocalizationLayer(dim::TwoDim, resolution::Tuple{Int, Int})
    x = imresize(build_grid(dim), resolution) ./ maximum(dim.x)
    return LocalizationLayer(x[:, :, :, :])
end

function (layer::LocalizationLayer)(x)
    return cat(
        x,
        repeat(layer.coords, 1, 1, 1, size(x, 4)),
        dims = 3
    )
end

function Waves.build_wave_encoder(;
        latent_dim::OneDim,
        h_size::Int,
        nfreq::Int,
        c0::Float32,
        k::Tuple{Int, Int} = (3, 3),
        in_channels::Int = 3,
        activation::Function = leakyrelu)

    nfields = 5

    return Chain(
        Waves.TotalWaveInput(),
        LocalizationLayer(env.dim, env.resolution),
        Waves.ResidualBlock(k, 2 + in_channels, 32, activation),
        Waves.ResidualBlock(k, 32, 64, activation),
        Waves.ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Parallel(
            vcat,
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
        ),
        b -> reshape(b, nfreq, nfields, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(
            x[:, [1], :],       # u_tot
            x[:, [2], :] ./ c0, # v_tot
            x[:, [3], :],       # u_inc
            x[:, [4], :] ./ c0, # v_inc
            x[:, [5], :]        # f
            )
        )
end

function Waves.AcousticEnergyModel(;
        env::WaveEnv, 
        latent_dim::OneDim,
        h_size::Int, 
        nfreq::Int, 
        pml_width::Float32,
        pml_scale::Float32)

    wave_encoder = Waves.build_wave_encoder(;
        latent_dim, 
        h_size, 
        nfreq,
        c0 = env.iter.dynamics.c0)

    mlp = Chain(
        Dense(length(vec(env.design)), h_size, leakyrelu), 
        Dense(h_size, h_size, leakyrelu), 
        Dense(h_size, h_size, leakyrelu),
        Dense(h_size, h_size, leakyrelu), 
        Dense(h_size, nfreq),
        SinWaveEmbedder(latent_dim, nfreq),
        c -> 2.0f0 * sigmoid.(c))

    design_encoder = Waves.DesignEncoder(env.design_space, mlp, env.integration_steps)
    F = Waves.SinusoidalSource(latent_dim, nfreq, env.source.freq)
    dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, pml_width, pml_scale)
    iter = Integrator(runge_kutta, dyn, env.dt)
    return AcousticEnergyModel(wave_encoder, design_encoder, F, iter, get_dx(latent_dim))
end

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)"
## declaring hyperparameters
h_size = 256
nfreq = 500
elements = 1024

horizon = 10
# horizon = 1

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
@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 1:500]
# @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 11:20]

## spliting data
idx = Int(round(length(data) * train_val_split))
train_data, val_data = data[1:idx], data[idx+1:end]

## preparing DataLoader(s)
train_loader = Flux.DataLoader(prepare_data(train_data, horizon); data_loader_kwargs...)
val_loader = Flux.DataLoader(prepare_data(val_data, horizon); data_loader_kwargs...)
println("Train Batches: $(length(train_loader)), Val Batches: $(length(val_loader))")

## contstruct model & train
latent_dim = OneDim(latent_gs, elements)
@time model = gpu(AcousticEnergyModel(;env, h_size, nfreq, pml_width, pml_scale, latent_dim))
# s, a, t, y = gpu(Flux.batch.(first(val_loader)))
# wave = model.wave_encoder(s)
# layer = gpu(LocalizationLayer(env.dim, env.resolution))
# model = gpu(BSON.load("tranable_source/checkpoint_step=2640/checkpoint.bson")[:model])
@time opt_state = Optimisers.setup(Optimisers.Adam(1f-4), model)

path = "localization_h_size=$(h_size)_latent_gs=$(latent_gs)_pml_width=$(pml_width)_nfreq=$nfreq"
model, opt_state = @time train!(model, opt_state;
    train_loader, 
    val_loader, 
    val_every,
    epochs,
    # path = path
    path = joinpath(DATA_PATH, path)
    )