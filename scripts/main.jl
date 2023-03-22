using Flux
using Flux.Losses: mse
using ReinforcementLearning
using CairoMakie
using Statistics: mean
using Serialization

using Waves

include("design_encoder.jl")
include("wave_net.jl")
include("inc_sc_wave_net.jl")

function train_wave_encoder_decoder_model!(
        opt::Flux.Optimise.AbstractOptimiser, 
        ps::Flux.Params,
        model,
        train_loader::Flux.DataLoader,
        test_loader::Flux.DataLoader,
        epochs::Int;
        path::String)
    
    comparison_path = mkpath(joinpath(path, "comparison"))

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
    end
end

train_data = deserialize("data/train")
test_data = deserialize("data/test")
train_loader = Flux.DataLoader(train_data, shuffle = true)
test_loader = Flux.DataLoader(test_data, shuffle = false)

grid_size = 4.0f0
elements = 256
dim = TwoDim(grid_size, elements)
design = train_data[1][1].design

# model_kwargs = Dict(
#     :fields => 6,
#     :h_fields => 32,
#     :z_fields => 2,
#     :activation => relu,
#     :design_size => 2 * length(vec(design)),
#     :h_size => 32,
#     :grid_size => 4.0f0,
#     :z_elements => prod(Int.(size(dim) ./ (2 ^ 3))))

# dim = TwoDim(grid_size, elements)
# dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

# model = gpu(IncScWaveNet(;model_kwargs..., dynamics_kwargs...))
# # model = gpu(WaveNet(;model_kwargs..., dynamics_kwargs...))

# opt = Adam(0.0001)
# ps = Flux.params(model)
# train_wave_encoder_decoder_model!(opt, ps, model, train_loader, test_loader, 2, path = "results/inc_sc_model")