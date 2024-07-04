using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

# dataset_name = "dataset_radii_design_space"
dataset_name = "dataset_pos_adjustment_masked"
DATA_PATH = "scratch/$dataset_name"
checkpoint = 9000
jobid = 42086
model_name = "AEM_batchsize=256_jobID=$jobid"
## generating paths
MODEL_PATH = joinpath(DATA_PATH, "models/$model_name/checkpoint_step=$checkpoint/checkpoint.bson")
## loading from storage
cnn_model = gpu(BSON.load(CNN_MODEL_PATH)[:model])

# for i in 495:497
## loading data
episode_number = 501 #i #497
ep = Episode(path = joinpath(DATA_PATH, "episodes/episode$episode_number.bson"))
horizon = 400
s, a, t, y = gpu(Flux.batch.(prepare_data(ep, horizon)))

## inferrence
@time cnn_y_hat = cpu(cnn_model(s[1, :], a[:, [1]], t[:, [1]]))
y = cpu(y)

## plotting comparison
t = cpu(t)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Variable Source Location Scattered Energy Prediction With Random Control")
lines!(ax, t[:, 1], y[:, 3, 1], label = "Ground Truth")
lines!(ax, t[:, 1], cnn_y_hat[:, 3, 1], color = (:red, 0.6), label = "Our Model")
axislegend(ax, position = :lt)
save("$(checkpoint)_$(episode_number)_$(horizon)_$jobid.png", fig)
# end

# using CSV, DataFrames, Statistics
# averaging = 5
# t = collect(1:length(y)) * horizon / 1000 / length(y)
# y = y[:, 3, 1]
# y_hat = cnn_y_hat[:, 3, 1]
# CSV.write("prediction_output.csv", DataFrame(["t" "y" "y_hat"], :auto))
# for i in 1:averaging:(length(t)-1)
#     i_step = [t[i] mean(y[i:i+averaging]) mean(y_hat[i:i+averaging])]
#     CSV.write("prediction_output.csv", DataFrame(i_step, :auto), append=true)
# end

# @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 468:500]
# data_loader_kwargs = Dict(:batchsize => 32, :shuffle => true, :partial => false)

# horizon = collect(20:10:200)
# our_error = Vector{Float32}[]
# # node_error = Vector{Float32}[]

# for h in horizon
#     val_loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
#     s, a, t, y = gpu(Flux.batch.(first(val_loader)))

#     y_sc = y[:, 3, :]
#     @time y_hat = our_model(s, a, t)[:, 3, :]
#     # @time y_hat = node_model(s, a, t)

#     error = cpu(vec(Flux.mse(y_sc, y_hat, agg = x -> Flux.mean(x, dims = 1))))
#     push!(our_error, error)
#     # push!(node_error, error)

#     BSON.bson("variable_source_results/our_error.bson", horizon = horizon, error = our_error)
#     # BSON.bson("variable_source_results/node_error.bson", horizon = horizon, error = node_error)
# end