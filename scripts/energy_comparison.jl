using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using ReinforcementLearning

Flux.CUDA.allowscalar(false)
Flux.device!(2)
display(Flux.device())
println("Loaded Packages")

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")
include("dataset.jl")
include("../src/model/node.jl")
include("../src/model/mpc.jl")

ep = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode500.bson")
horizon = 200
s, a, t, y = gpu(get_energy_data(ep, horizon, 1))
y = cpu(y)

# pml_checkpoint = 4680
# model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal/checkpoint_step=$pml_checkpoint/checkpoint.bson")[:model])
# @time y_hat_pml = cpu(model([s], a[:, :], t[:, :]))
y_hat_pml = BSON.load("rebuttal/pml_prediction.bson")[:y_hat]

# no_pml_checkpoint = 2300
# no_pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_no_pml/checkpoint_step=$no_pml_checkpoint/checkpoint.bson")[:model])
# @time y_hat_no_pml = cpu(no_pml_model([s], a[:, :], t[:, :]))
# BSON.bson("no_pml_prediction.bson", y_hat = y_hat_no_pml)
y_hat_no_pml = BSON.load("rebuttal/no_pml_prediction.bson")[:y_hat]

# node_checkpoint = 2300
# node_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_node/checkpoint_step=$node_checkpoint/checkpoint.bson")[:model])
# @time y_hat_node = cpu(node_model([s], a[:, :], t[:, :]))
# BSON.bson("node_prediction.bson", y_hat = y_hat_node)
y_hat_node = BSON.load("rebuttal/node_prediction.bson")[:y_hat]

t = cpu(t)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Scattered Energy Prediction with Random Control")
CairoMakie.ylims!(ax, 0.0, 50.0)
lines!(ax, t, y[:, 3], color = :blue, label = "Ground Truth")
lines!(ax, t, y_hat_pml[:, 3, 1], color = (:green, 0.6), label = "Ours (PML)")
lines!(ax, t, y_hat_no_pml[:, 3, 1], color = (:orange, 0.6), label = "Ours (No PML)")
lines!(ax, t, y_hat_node[:, 1], color = (:red, 0.6), label = "NeuralODE")
axislegend(ax, position = :lt)
save("energy_comparison_full.png", fig)
