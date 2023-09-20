using Waves, Flux, Optimisers
Flux.CUDA.allowscalar(false)
using CairoMakie
using BSON
include("model.jl")

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")
length(ep)

horizon = 3
batchsize = 1
h_size = 256
nfreq = 50

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))

latent_dim = OneDim(30.0f0, 1024)
dx = get_dx(latent_dim)
wave_encoder = gpu(build_wave_encoder(;latent_dim, nfreq))

dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

z0 = wave_encoder(s)