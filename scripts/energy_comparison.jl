using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using ReinforcementLearning

Flux.CUDA.allowscalar(false)
Flux.device!(0)
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
horizon = 100

idx = 1
# s, a, t, y = gpu(get_energy_data(ep, horizon, idx))
s, a, t, y = prepare_data(ep, horizon)
s, a, t, y = gpu((s[idx], a[idx], t[idx], y[idx]))
y = cpu(y)

checkpoint_step = 2300
pml_model = gpu(BSON.load("/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/validate_pml_model_new_code_old_dataloader/checkpoint_step=$(checkpoint_step)/checkpoint.bson")[:model])
# no_pml_model = gpu(BSON.load("../AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_no_pml/checkpoint_step=$(checkpoint_step)/checkpoint.bson")[:model])

# z_pml = cpu(generate_latent_solution(pml_model, [s], a[:, :], t[:, :]))
# z_pml_sc = (z_pml[:, 1, 1, :] .- z_pml[:, 3, 1, :]) .^ 2

# z_no_pml = cpu(generate_latent_solution(no_pml_model, [s], a[:, :], t[:, :]))
# z_no_pml_sc = (z_no_pml[:, 1, 1, :] .- z_no_pml[:, 3, 1, :]) .^ 2
# dim = cpu(pml_model.iter.dynamics.dim)
# t = cpu(t)

# fig = Figure()
# ax1 = Axis(fig[1, 1], xlabel = "Space (m)", ylabel = "Time (s)", title = "Scattered Energy Field (PML)")
# ax2 = Axis(fig[1, 2], xlabel = "Space (m)", ylabel = "Time (s)", title = "Scattered Energy Field (No PML)")
# heatmap!(ax1, dim.x, t, z_pml_sc, colormap = :inferno, colorrange = (0.0, 1.0))
# heatmap!(ax2, dim.x, t, z_no_pml_sc, colormap = :inferno, colorrange = (0.0, 1.0))
# save("latent.png", fig)


@time y_hat_pml = cpu(pml_model([s], a[:, :], t[:, :]))

node_model = gpu(BSON.load("/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/validate_node_model_new_code_old_dataloader/checkpoint_step=$(checkpoint_step)/checkpoint.bson")[:model])
@time y_hat_node = cpu(node_model([s], a[:, :], t[:, :]))

t = cpu(t)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Scattered Energy Prediction with Random Control")
lines!(ax, t, y[:, 3], color = :blue, label = "Ground Truth")
lines!(ax, t, y_hat_pml[:, 3], color = (:green, 0.6), label = "Ours (PML)")
lines!(ax, t, y_hat_node[:, 1], color = (:red, 0.6), label = "NeuralODE")
axislegend(ax, position = :lt)
# save("prediction_new_code_step=$(checkpoint_step).png", fig)
save("test.png", fig)
















# # pml_checkpoint = 4680
# # model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal/checkpoint_step=$pml_checkpoint/checkpoint.bson")[:model])
# # @time y_hat_pml = cpu(model([s], a[:, :], t[:, :]))
# y_hat_pml = BSON.load("rebuttal/pml_prediction.bson")[:y_hat]

# # no_pml_checkpoint = 2300
# # no_pml_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_no_pml/checkpoint_step=$no_pml_checkpoint/checkpoint.bson")[:model])
# # @time y_hat_no_pml = cpu(no_pml_model([s], a[:, :], t[:, :]))
# # BSON.bson("no_pml_prediction.bson", y_hat = y_hat_no_pml)
# y_hat_no_pml = BSON.load("rebuttal/no_pml_prediction.bson")[:y_hat]

# # node_checkpoint = 2300
# # node_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_node/checkpoint_step=$node_checkpoint/checkpoint.bson")[:model])
# # @time y_hat_node = cpu(node_model([s], a[:, :], t[:, :]))
# # BSON.bson("node_prediction.bson", y_hat = y_hat_node)
# y_hat_node = BSON.load("rebuttal/node_prediction.bson")[:y_hat]

# t = cpu(t)
# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Scattered Energy Prediction with Random Control")
# CairoMakie.ylims!(ax, 0.0, 50.0)
# lines!(ax, t, y[:, 3], color = :blue, label = "Ground Truth")
# lines!(ax, t, y_hat_pml[:, 3, 1], color = (:green, 0.6), label = "Ours (PML)")
# lines!(ax, t, y_hat_no_pml[:, 3, 1], color = (:orange, 0.6), label = "Ours (No PML)")
# lines!(ax, t, y_hat_node[:, 1], color = (:red, 0.6), label = "NeuralODE")
# axislegend(ax, position = :lt)
# save("energy_comparison_full.png", fig)
