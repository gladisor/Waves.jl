using Waves, Flux, Optimisers
Flux.CUDA.allowscalar(false)
using CairoMakie
using BSON
include("model.jl")

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")
length(ep)

horizon = 20
batchsize = 1
h_size = 256
nfreq = 50

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))

latent_dim = OneDim(30.0f0, 1024)
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
emb = gpu(
    Chain(
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0)
        ))

freq_coefs = gpu(randn(Float32, nfreq, 4, 1))
z0 = emb(freq_coefs)

loss, back = Flux.pullback(z0) do _z0
    z = iter(_z0, t, θ)
    e = latent_energy(z, dx)
    return Flux.mse(e, y)
end

@time gs_AD = cpu(back(one(loss)))[1]
z = iter(z0, t, θ)

loss, back = Flux.pullback(z) do _z
    e = latent_energy(_z, dx)
    return Flux.mse(e, y)
end

adj = back(one(loss))[1]
@time gs_AS = cpu(adjoint_sensitivity(iter, z, t, θ, adj))

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = "Space (m)", ylabel = "Autodiff Gradient", title = "Gradient of energy difference with respect to initial total displacement")
ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel = "Adjoint Sensitivity Gradient")

lines!(ax1, latent_dim.x, gs_AD[:, 1, 1], color = :blue)
lines!(ax2, latent_dim.x, gs_AS[:, 1, 1], color = :orange)
save("gs.png", fig)

z = iter(z0, t, θ)
z = cpu(z)
tspan = cpu(vec(t))
render(latent_dim, z[:, :, 1, :], tspan, path = "latent.mp4")