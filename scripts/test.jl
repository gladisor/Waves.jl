using Waves, CairoMakie, Flux, BSON
using Optimisers
Flux.CUDA.allowscalar(false)
Flux.device!(0)
display(Flux.device())
include("model_modifications.jl")

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/"
ENV_PATH = joinpath(DATA_PATH, "env.bson")
EPISODE_PATH = joinpath(DATA_PATH, "episodes/episode500.bson")

step = 2300
MODEL_PATH = joinpath(DATA_PATH, "trainable_pml_localization_horizon=20_batchsize=32_h_size=256_latent_gs=100.0_pml_width=10.0_nfreq=500/checkpoint_step=$step/checkpoint.bson")

env = BSON.load(ENV_PATH)[:env]
ep = Episode(path = EPISODE_PATH)
model = gpu(BSON.load(MODEL_PATH)[:model])

s, a, t, y = prepare_data(ep, 200)
idx = 1
s, a, t, y = gpu(Flux.batch.((s[[idx]], a[[idx]], t[[idx]], y[[idx]])))
@time y_hat = cpu(model(s, a, t))

y = cpu(y)
t = cpu(vec(t))

fig = Figure()
ax = Axis(fig[1, 1], title = "Scattered Energy", xlabel = "Time (s)", ylabel = "Energy")
lines!(ax, t[:, 1], y[:, 3, 1], color = :blue, label = "True")
lines!(ax, t[:, 1], y_hat[:, 3, 1], color = (:orange, 0.5), label = "Predicted")
axislegend(ax, position = :rb)
save("$(step)_sc.png", fig)

fig = Figure()
ax = Axis(fig[1, 1], title = "Incident Energy", xlabel = "Time (s)", ylabel = "Energy")
lines!(ax, t[:, 1], y[:, 2, 1], color = :blue, label = "True")
lines!(ax, t[:, 1], y_hat[:, 2, 1], color = (:orange, 0.5), label = "Predicted")
axislegend(ax, position = :rb)
save("$(step)_inc.png", fig)

fig = Figure()
ax = Axis(fig[1, 1], title = "Total Energy", xlabel = "Time (s)", ylabel = "Energy")
lines!(ax, t[:, 1], y[:, 1, 1], color = :blue, label = "True")
lines!(ax, t[:, 1], y_hat[:, 1, 1], color = (:orange, 0.5), label = "Predicted")
axislegend(ax, position = :rb)
save("$(step)_tot.png", fig)