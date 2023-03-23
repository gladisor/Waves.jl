using Flux
using Flux.Losses: mse
using ReinforcementLearning
using CairoMakie
using Statistics: mean
using BSON
using JLD2

using Waves

include("design_encoder.jl")
include("wave_net.jl")
include("inc_sc_wave_net.jl")
include("latent_wave_separation.jl")

function train_wave_encoder_decoder_model!(
        opt::Flux.Optimise.AbstractOptimiser, 
        ps::Flux.Params,
        model,
        train_loader::Flux.DataLoader,
        test_loader::Flux.DataLoader,
        epochs::Int;
        path::String)
    
    comparison_path = mkpath(joinpath(path, "comparison"))
    model_path = mkpath(joinpath(path, "model"))

    train_loss_history = Float32[]
    test_loss_history = Float32[]

    for epoch in 1:epochs

        train_epoch_loss = Float32[]

        ## Training logic
        Flux.train!(ps, train_loader, opt) do s, a

            ## Grab a sample from data loader and compute loss
            train_loss = loss(model, gpu(s[1]), gpu(a[1]))

            ## Log loss & reset model dynamics
            Flux.ignore() do
                push!(train_epoch_loss, train_loss)
                Waves.reset!(model)
            end

            return train_loss
        end

        test_epoch_loss = Float32[]

        for (i, (s, a)) in enumerate(test_loader)
            s, a = gpu(s[1]), gpu(a[1])
            push!(test_epoch_loss, loss(model, s, a))
            plot_comparison!(model, s, a, path = joinpath(comparison_path, "comparison_$i.png"))
            Waves.reset!(model)
        end

        ## Storing loss data for history
        push!(train_loss_history, mean(train_epoch_loss))
        push!(test_loss_history, mean(test_epoch_loss))
        println("Epoch: $epoch, Train Loss: $(train_loss_history[end]), Test Loss: $(test_loss_history[end])")
        plot_loss!(train_loss_history, test_loss_history, path = joinpath(path, "train_loss.png"))
        BSON.@save joinpath(model_path, "model$epoch.bson") model
    end
end

function load_wave_data(path::String)

    s = WaveEnvState[]
    a = AbstractDesign[]

    for file_path in readdir(path, join = true)
        jldopen(file_path) do file
            println(file)
            push!(s, file["s"])
            push!(a, file["a"])
        end
    end

    return (s, a)
end

train_data = load_wave_data("data/train");
test_data = load_wave_data("data/test");

first_state = first(train_data[1])
dim = first_state.sol.total.dim
elements = size(dim)[1]
grid_size = maximum(dim.x)
design = first_state.design

train_loader = Flux.DataLoader(train_data, shuffle = true)
test_loader = Flux.DataLoader(test_data, shuffle = false)

design_size = 2 * length(vec(design))
z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))

model_kwargs = Dict(:fields => 6, :h_fields => 128, :z_fields => 2, :activation => relu, :design_size => design_size, :h_size => 256, :grid_size => 4.0f0, :z_elements => z_elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

# model = gpu(IncScWaveNet(;model_kwargs..., dynamics_kwargs...))
model = gpu(WaveNet(;model_kwargs..., dynamics_kwargs...))
# model = gpu(LatentWaveSeparation(;model_kwargs..., dynamics_kwargs...))

lr = 0.001
opt = Adam(lr)
ps = Flux.params(model)

model_name = String(Symbol(typeof(model)))
train_wave_encoder_decoder_model!(opt, ps, model, train_loader, test_loader, 100, path = "results/scattered_$(model_name)_lr=$lr")