using Waves, CairoMakie, Flux, BSON
using Optimisers
Flux.CUDA.allowscalar(false)
Flux.device!(2)
display(Flux.device())
include("model_modifications.jl")

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/"
ENV_PATH = joinpath(DATA_PATH, "env.bson")
EPISODE_PATH = joinpath(DATA_PATH, "episodes/episode500.bson")
MODEL_PATH = joinpath(DATA_PATH, "localization_h_size=256_latent_gs=100.0_pml_width=10.0_nfreq=500/checkpoint_step=6900/checkpoint.bson")
# MODEL_PATH = joinpath(DATA_PATH, "localization_horizon=20_batchsize=32_h_size=256_latent_gs=100.0_pml_width=10.0_nfreq=500/checkpoint_step=2380/checkpoint.bson")

env = BSON.load(ENV_PATH)[:env]
ep = Episode(path = EPISODE_PATH)
model = gpu(BSON.load(MODEL_PATH)[:model])

s, a, t, y = prepare_data(ep, 10)
idx = 19
s, a, t, y = gpu(Flux.batch.((s[[idx]], a[[idx]], t[[idx]], y[[idx]])))
# @time y_hat = cpu(model(s, a, t))
# Waves.generate_latent_solution(model, s, a, t)

# y = cpu(y)
# t = cpu(vec(t))

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, t[:, 1], y[:, 3, 1], color = :blue)
# lines!(ax, t[:, 1], y_hat[:, 3, 1], color = :orange)
# save("pred.png", fig)
