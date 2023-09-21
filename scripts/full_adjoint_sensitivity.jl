using Waves, Flux, Optimisers
Flux.CUDA.allowscalar(false)
using ChainRulesCore
using CairoMakie
using BSON
include("model.jl")

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")

horizon = 3
batchsize = 1
h_size = 256
nfreq = 50

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))
tspan = cpu(vec(t))

latent_dim = OneDim(15.0f0, 1024)
dx = get_dx(latent_dim)

mlp = Chain(
    Dense(18, h_size, leakyrelu), 
    Dense(h_size, h_size, leakyrelu), 
    Dense(h_size, nfreq),
    SinWaveEmbedder(latent_dim, nfreq),
    c -> 2.0f0 * sigmoid.(c))
design_encoder = gpu(DesignEncoder(env.design_space, mlp, env.integration_steps))

C = design_encoder(s, a, t)
F = gpu(SinusoidalSource(latent_dim, nfreq, env.source.freq))
θ = [C, F]

dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

c0 = env.iter.dynamics.c0
emb = gpu(Chain(SinWaveEmbedder(latent_dim, nfreq), x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0)))  
freq_coefs = gpu(randn(Float32, nfreq, 4, 1))




z = iter(emb(freq_coefs), t, θ)
energy = cpu(latent_energy(z, dx))
plot_energy(tspan, energy[:, :, 1], path = "pre_opt_latent_energy.png")
plot_energy(tspan, cpu(y[:, :, 1]), path = "true_energy.png")




x = [freq_coefs, θ]
opt_state = Optimisers.setup(Optimisers.Descent(1e-2), x)

for i in 1:20
    loss, back = Flux.pullback(x) do _x
        z0 = emb(_x[1])
        z = iter(z0, t, _x[2])
        e = latent_energy(z, dx)
        return Flux.mse(e, y)
    end
    gs = back(one(loss))[1]

    opt_state, x = Optimisers.update(opt_state, x, gs)
    println(loss)
end




z = iter(emb(freq_coefs), t, θ)
energy = cpu(latent_energy(z, dx))
plot_energy(tspan, energy[:, :, 1], path = "post_opt_latent_energy.png")