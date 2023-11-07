using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
using ReinforcementLearning

println("Loaded Packages")
Flux.device!(2)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")
include("dataset.jl")
include("../src/model/node.jl")
include("../src/model/mpc.jl")

@time data = [Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode$i.bson") for i in 468:500]
data_loader_kwargs = Dict(:batchsize => 32, :shuffle => true, :partial => false)

pml_checkpoint = 2300
no_pml_checkpoint = 2300
node_checkpoint = 2300

# pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal/checkpoint_step=$pml_checkpoint/checkpoint.bson")[:model])
# no_pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_no_pml/checkpoint_step=$no_pml_checkpoint/checkpoint.bson")[:model])
node_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_node/checkpoint_step=$node_checkpoint/checkpoint.bson")[:model])

# val_loader = Flux.DataLoader(EpisodeDataset(data, 5, get_energy_data); data_loader_kwargs...)
# s, a, t, y = gpu(first(val_loader))
# z = cpu(generate_latent_solution(node_model, s, a, t))

# tspan = cpu(t[:, 1])

# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, z[:, 1, 1, :], colormap = :ice)
# save("node_latent.png", fig)

horizon = collect(20:10:200)
node_error = Vector{Float32}[]
for h in horizon
    val_loader = Flux.DataLoader(EpisodeDataset(data, h, get_energy_data); data_loader_kwargs...)
    s, a, t, y = gpu(first(val_loader))

    # energy_pml = pml_model(s, a, t)[:, 3, :]
    # energy_no_pml = no_pml_model(s, a, t)[:, 3, :]
    @time energy_node = node_model(s, a, t)
    y_sc = y[:, 3, :]

    # pml_error = cpu(vec(Flux.mse(y_sc, energy_pml, agg = x -> Flux.mean(x, dims = 1))))
    # no_pml_error = cpu(vec(Flux.mse(y_sc, energy_no_pml, agg = x -> Flux.mean(x, dims = 1))))
    error = cpu(vec(Flux.mse(y_sc, energy_node, agg = x -> Flux.mean(x, dims = 1))))
    push!(node_error, error)
    BSON.bson("node_error.bson", horizon = horizon, error = node_error)
end