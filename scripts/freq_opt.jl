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
nfreq = 200

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))

latent_dim = OneDim(30.0f0, 1024)
dx = get_dx(latent_dim)
dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

c = gpu(ones(Float32, size(latent_dim)..., batchsize))
C = t -> c
# F = gpu(SinusoidalSource(latent_dim, nfreq, env.source.freq))
F = gpu(GaussianSource(latent_dim, [0.0f0], [0.3f0], [1.0f0], env.source.freq))
θ = [C, F]

freq_coefs = gpu(randn(Float32, nfreq, 4, batchsize))

c0 = WATER
emb = gpu(
    Chain(
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0)
        )
    )

z0 = emb(freq_coefs)
z = iter(z0, t, θ)

render(latent_dim, cpu(z[:, :, 1, :]), cpu(t[:, 1]), path = "optimization/latent_pre_opt.mp4")
energy = cpu(latent_energy(z, dx))
true_energy = cpu(y)
tspan = vec(cpu(t))
plot_energy(tspan, energy[:, :, 1], path = "optimization/latent_energy_pre_opt.png")
plot_energy(tspan, true_energy[:, :, 1], path = "optimization/true_energy.png")
# opt_state = Optimisers.setup(Optimisers.Adam(1e-2), freq_coefs)
opt_state = Optimisers.setup(Optimisers.Descent(1e-3), freq_coefs)

for i in 1:20
    z0, back1 = Flux.pullback(freq_coefs) do _freq_coefs
        return emb(_freq_coefs)
    end

    z, back2 = Flux.pullback(z0) do _z0
        return iter(_z0, t, θ)
    end

    loss, back3 = Flux.pullback(z) do _z
        energy = latent_energy(_z, dx)
        return Flux.mse(energy, y)
    end

    println("Iteration: $i")
    println("Loss: $loss")

    adj = back3(one(loss))[1]
    # @time gs_z = adjoint_sensitivity(iter, z, t, θ, adj)
    @time gs_z = back2(adj ./ horizon)[1]
    gs_freq_coef = back1(gs_z)[1]
    opt_state, freq_coefs = Optimisers.update(opt_state, freq_coefs, gs_freq_coef)
end

# # fig = Figure()
# # ax1 = Axis(fig[1, 1])
# # lines!(ax1, latent_dim.x, gs[:, 1, 1])
# # lines!(ax1, latent_dim.x, gs[:, 3, 1])
# # ax2 = Axis(fig[2, 1])
# # lines!(ax2, latent_dim.x, gs[:, 2, 1])
# # lines!(ax2, latent_dim.x, gs[:, 4, 1])
# # save("gs.png", fig)


z0 = emb(freq_coefs)
z = iter(z0, t, θ)
energy = cpu(latent_energy(z, dx))
plot_energy(tspan, energy[:, :, 1], path = "optimization/latent_energy_post_opt.png")
