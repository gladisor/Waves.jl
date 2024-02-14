using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
checkpoint = 10040 #10500
# our_model_name = "horizon=20,lr=0.0001"
our_model_name = "ours_balanced_field_scale"
node_model_name = "node_horizon=20,lr=0.0001"
pinn_model_name = "wave_control_pinn_accumulate=8"
## generating paths
OUR_MODEL_PATH = joinpath(DATA_PATH, "models/$our_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
NODE_MODEL_PATH = joinpath(DATA_PATH, "models/$node_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
PINN_MODEL_PATH = joinpath(DATA_PATH, "models/$pinn_model_name/checkpoint_step=$(checkpoint*8)/checkpoint.bson")

## loading from storage
our_model = gpu(BSON.load(OUR_MODEL_PATH)[:model])
node_model = gpu(BSON.load(NODE_MODEL_PATH)[:model])
pinn_model = gpu(BSON.load(PINN_MODEL_PATH)[:model])

# for i in 495:497
#     ## loading data
#     # episode_number = 492 #i #497
#     episode_number = i

#     ep = Episode(path = joinpath(DATA_PATH, "episodes/episode$episode_number.bson"))
#     horizon = 100
#     s, a, t, y = gpu(Flux.batch.(prepare_data(ep, horizon)))
#     ## inferrence

#     idx = 100
#     @time our_y_hat = cpu(our_model(s[idx, :], a[:, [idx]], t[:, [idx]]))
#     # BSON.bson("variable_source_results/ours.bson", y_hat = our_y_hat[:, 3, 1])
#     @time node_y_hat = cpu(node_model(s[idx, :], a[:, [idx]], t[:, [idx]]))
#     # BSON.bson("variable_source_results/node.bson", y_hat = node_y_hat[:, 1, 1])
#     @time pinn_y_hat = cpu(pinn_model(s[idx, :], a[:, [idx]], t[:, [idx]]))
#     # BSON.bson("results/variable_source_results/pinn.bson", y_hat = pinn_y_hat[:, 3, 1])
#     y = cpu(y)
#     # BSON.bson("variable_source_results/ground_truth.bson", y = y[:, 3, 1])

#     ## plotting comparison
#     t = cpu(t)
#     fig = Figure()
#     ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Variable Source Location Scattered Energy Prediction With Random Control")
#     lines!(ax, t[:, idx], y[:, 3, idx], label = "Ground Truth")
#     lines!(ax, t[:, idx], our_y_hat[:, 3, 1], color = (:green, 0.6), label = "Ours (PML)")
#     lines!(ax, t[:, idx], node_y_hat[:, 1, 1], color = (:red, 0.6), label = "NeuralODE")
#     lines!(ax, t[:, idx], pinn_y_hat[:, 3, 1], color = (:purple, 0.6), label = "PINC")
#     axislegend(ax, position = :lt)
#     save("$dataset_name,_checkpoint=$checkpoint,episode=$episode_number.png", fig)
#     # save("variable_source_results/$dataset_name,_checkpoint=$checkpoint,episode=$episode_number.png", fig)
#     # save("results.png", fig)
# end

@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 468:500]
# data_loader_kwargs = Dict(:batchsize => 32, :shuffle => true, :partial => false)
data_loader_kwargs = Dict(:batchsize => 4, :shuffle => true, :partial => false)

horizon = collect(20:10:200)
# # our_error = Vector{Float32}[]
# # node_error = Vector{Float32}[]

# pinn_error = BSON.load("results/variable_source_results/pinn_error.bson")[:error]
pinn_error = Vector{Float32}[]

accumulate = 8

for h in horizon
    val_loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)

    error = []

    for i in 1:accumulate
        s, a, t, y = gpu(Flux.batch.(first(val_loader)))
        y_sc = y[:, 3, :]
        @time y_hat = pinn_model(s, a, t)[:, 3, :]
        error = vcat(
            error,
            cpu(vec(Flux.mse(y_sc, y_hat, agg = x -> Flux.mean(x, dims = 1))))
        )
    end

    push!(pinn_error, error)
end

BSON.bson("pinn_error.bson", horizon = horizon, error = pinn_error)







# for h in horizon
#     val_loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
#     s, a, t, y = gpu(Flux.batch.(first(val_loader)))

#     y_sc = y[:, 3, :]
#     @time y_hat = pinn_model(s, a, t)[:, 3, :]
#     @time y_hat = our_model(s, a, t)[:, 3, :]
#     @time y_hat = node_model(s, a, t)

#     error = cpu(vec(Flux.mse(y_sc, y_hat, agg = x -> Flux.mean(x, dims = 1))))
#     push!(pinn_error, error)
#     push!(our_error, error)
#     push!(node_error, error)
    
#     BSON.bson("results/variable_source_results/pinn_error.bson", horizon = horizon, error = pinn_error)
#     BSON.bson("variable_source_results/our_error.bson", horizon = horizon, error = our_error)
#     BSON.bson("variable_source_results/node_error.bson", horizon = horizon, error = node_error)
# end

