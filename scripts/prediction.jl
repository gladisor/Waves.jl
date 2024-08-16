using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
println("Loaded Packages")

# dataset_name = "dataset_radii_design_space"
# dataset_name = "dataset_pos_adjustment_masked"
dataset_name = "pos_adjustment_masked_M=2"
DATA_PATH = "scratch/$dataset_name"
checkpoint = 10500
jobid = 42476
model_name = "AEM_batchsize=64_jobID=$jobid"
## generating paths
MODEL_PATH = joinpath(DATA_PATH, "models/$model_name/checkpoint_step=$checkpoint/checkpoint.bson")
## loading from storage
model = BSON.load(MODEL_PATH)[:model]

for s_i in [1 51]
    for i in 497:499
        ## loading data
        episode_number = i #497
        ep = Episode(path = joinpath(DATA_PATH, "episodes/episode$episode_number.bson"))
        horizon = 100
        s, a, t, y = Flux.batch.(prepare_data(ep, horizon))
        start_index = s_i
        s = s[start_index:end]
        a = a[:, start_index:end]
        t = t[:, start_index:end]
        y = y[:, :, start_index:end]
        ## inferrence
        @time y_hat = model(s[1, :], a[:, [1]], t[:, [1]])

        ## plotting comparison
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Variable Source Location Scattered Energy Prediction With Random Control")
        lines!(ax, t[:, 1], y[:, 3, 1], label = "Ground Truth")
        lines!(ax, t[:, 1], y_hat[:, 3, 1], color = (:red, 0.6), label = "Our Model")
        axislegend(ax, position = :lt)
        save(joinpath(mkpath("$(jobid)_prediction"), "$(checkpoint)_$(episode_number)_$(horizon)_$(jobid)_$(start_index).png"), fig)
        println("saved: $(checkpoint)_$(episode_number)_$(horizon)_$(jobid)_$(start_index).png")
    end
end
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