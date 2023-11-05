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

env = BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/env.bson")[:env]

pml_checkpoint = 2920
pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal/checkpoint_step=$pml_checkpoint/checkpoint.bson")[:model])
# no_pml_checkpoint = 2060
# no_pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_no_pml/checkpoint_step=$no_pml_checkpoint/checkpoint.bson")[:model])
node_checkpoint = 980
node_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_node/checkpoint_step=$node_checkpoint/checkpoint.bson")[:model])

policy = RandomDesignPolicy(action_space(env))

horizon = 200
shots = 1
alpha = 1.0

ep = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode500.bson")
s, a, t, y = gpu(get_energy_data(ep, horizon, 1))

@time y_hat_pml = cpu(pml_model([s], a[:, :], t[:, :]))
# @time y_hat_no_pml = cpu(no_pml_model([s], a[:, :], t[:, :]))
@time y_hat_node = cpu(node_model([s], a[:, :], t[:, :]))

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy")
lines!(ax, cpu(t), cpu(y[:, 3]), color = :blue, label = "Ground Truth")
lines!(ax, cpu(t), y_hat_pml[:, 3, 1], color = (:orange, 0.6), label = "Ours (PML)")
# lines!(ax, cpu(t), y_hat_no_pml[:, 3, 1], color = (:green, 0.6), label = "Ours (No PML)")
lines!(ax, cpu(t), y_hat_node[:, 1], color = (:purple, 0.6), label = "Node")
axislegend(ax, position = :lt)
save("energy_comparison.png", fig)