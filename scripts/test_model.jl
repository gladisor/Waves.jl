using Waves, Flux, BSON
using Images: imresize
using CairoMakie
include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")

DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/"
ENV_PATH = joinpath(DATA_PATH, "env.bson")
EPISODE_PATH = joinpath(DATA_PATH, "episodes/episode500.bson")

env = BSON.load(ENV_PATH)[:env]
ep = Episode(path = EPISODE_PATH)

data = Flux.DataLoader(prepare_data(ep, 3), shuffle = true, batchsize = 2)
s, a, t, y = gpu(Flux.batch.(first(data)))

h_size = 256
activation = leakyrelu
nfreq = 500

pml_width = 10.0f0
pml_scale = 10000.0f0

latent_dim = OneDim(100.0f0, 1024)
wave_encoder = WaveEncoder(env, h_size, activation, nfreq, latent_dim)
design_encoder = DesignEncoder(env, h_size, leakyrelu, 10, latent_dim)
dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, pml_width, pml_scale)
iter = Integrator(runge_kutta, dyn, env.dt)

model = gpu(AcousticEnergyModel(wave_encoder, design_encoder, iter, get_dx(latent_dim), env.source.freq))


loss, back = Flux.pullback(m -> Flux.mse(m(s, a, t), y), model)
println(loss)
gs = back(one(loss))


# render_latent_solution(model, s, a, t)
# y_hat = model(s, a, t)
# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, y_hat[:, 1, 1])
# lines!(ax, y_hat[:, 2, 1])
# lines!(ax, y_hat[:, 3, 1])
# save("energy.png", fig)