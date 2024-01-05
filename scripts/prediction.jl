using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(0)
display(Flux.device())
include("random_pos_gaussian_source.jl")
include("node.jl")

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
checkpoint = 10500
our_model_name = "horizon=20,lr=0.0001"
node_model_name = "node_horizon=20,lr=0.0001"
## generating paths
OUR_MODEL_PATH = joinpath(DATA_PATH, "models/$our_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
NODE_MODEL_PATH = joinpath(DATA_PATH, "models/$node_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
## loading from storage
our_model = gpu(BSON.load(OUR_MODEL_PATH)[:model])
node_model = gpu(BSON.load(NODE_MODEL_PATH)[:model])

# # for i in 495:497
#     ## loading data
#     episode_number = 492 #i #497
#     ep = Episode(path = joinpath(DATA_PATH, "episodes/episode$episode_number.bson"))
#     horizon = 200
#     s, a, t, y = gpu(Flux.batch.(prepare_data(ep, horizon)))

#     ## inferrence
#     @time our_y_hat = cpu(our_model(s[1, :], a[:, [1]], t[:, [1]]))
#     BSON.bson("variable_source_results/ours.bson", y_hat = our_y_hat[:, 3, 1])
#     @time node_y_hat = cpu(node_model(s[1, :], a[:, [1]], t[:, [1]]))
#     BSON.bson("variable_source_results/node.bson", y_hat = node_y_hat[:, 1, 1])
#     y = cpu(y)
#     BSON.bson("variable_source_results/ground_truth.bson", y = y[:, 3, 1])

#     ## plotting comparison
#     t = cpu(t)
#     fig = Figure()
#     ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Variable Source Location Scattered Energy Prediction With Random Control")
#     lines!(ax, t[:, 1], y[:, 3, 1], label = "Ground Truth")
#     lines!(ax, t[:, 1], our_y_hat[:, 3, 1], color = (:green, 0.6), label = "Ours (PML)")
#     lines!(ax, t[:, 1], node_y_hat[:, 1, 1], color = (:red, 0.6), label = "NeuralODE")
#     axislegend(ax, position = :lt)
#     save("variable_source_results/$dataset_name,_checkpoint=$checkpoint,episode=$episode_number.png", fig)
# # end

@time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 468:500]
data_loader_kwargs = Dict(:batchsize => 32, :shuffle => true, :partial => false)

horizon = collect(20:10:200)
our_error = Vector{Float32}[]
# node_error = Vector{Float32}[]

for h in horizon
    val_loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
    s, a, t, y = gpu(Flux.batch.(first(val_loader)))

    y_sc = y[:, 3, :]
    @time y_hat = our_model(s, a, t)[:, 3, :]
    # @time y_hat = node_model(s, a, t)

    error = cpu(vec(Flux.mse(y_sc, y_hat, agg = x -> Flux.mean(x, dims = 1))))
    push!(our_error, error)
    # push!(node_error, error)

    BSON.bson("variable_source_results/our_error.bson", horizon = horizon, error = our_error)
    # BSON.bson("variable_source_results/node_error.bson", horizon = horizon, error = node_error)
end

