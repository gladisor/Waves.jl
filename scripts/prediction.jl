using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(2)
display(Flux.device())
include("random_pos_gaussian_source.jl")

checkpoint = 7040
dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
name = "horizon=20,lr=0.0001"
MODEL_PATH = joinpath(DATA_PATH, "models/$name/checkpoint_step=$checkpoint/checkpoint.bson")
model = gpu(BSON.load(MODEL_PATH)[:model])

episode_number = 499
ep = Episode(path = joinpath(DATA_PATH, "episodes/episode$episode_number.bson"))

horizon = 200
s, a, t, y = gpu(Flux.batch.(prepare_data(ep, horizon)))
y = cpu(y)
@time y_hat = cpu(model(s[1, :], a[:, [1]], t[:, [1]]))

t = cpu(t)
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, t[:, 1], y[:, 3, 1])
lines!(ax, t[:, 1], y_hat[:, 3, 1], color = (:orange, 0.6))
save("$name,_checkpoint=$checkpoint,episode=$episode_number.png", fig)



# @time data = [Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode$i.bson") for i in 468:500]
# data_loader_kwargs = Dict(:batchsize => 32, :shuffle => true, :partial => false)


# BSON.bson("pml_prediction.bson", y_hat = y_hat)
# # horizon = collect(20:10:200)
# horizon = collect(20:10:200)
# pml_error = Vector{Float32}[]

# for h in horizon
#     val_loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
#     s, a, t, y = gpu(Flux.batch.(first(val_loader)))

#     y_sc = y[:, 3, :]
#     @time y_hat = model(s, a, t)[:, 3, :]
#     error = cpu(vec(Flux.mse(y_sc, y_hat, agg = x -> Flux.mean(x, dims = 1))))
#     push!(pml_error, error)

#     BSON.bson("pml_error.bson", horizon = horizon, error = pml_error)
# end

