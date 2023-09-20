using Waves
using Flux
Flux.CUDA.allowscalar(false)
using Optimisers
using CairoMakie
using BSON

include("model.jl")

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")
length(ep)

horizon = 3
batchsize = 1
h_size = 256
nfreq = 100

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))
tspan = vec(cpu(t))
true_energy = cpu(y)

latent_dim = OneDim(30.0f0, 1024)
dx = get_dx(latent_dim)

wave_encoder = build_wave_encoder(;latent_dim, nfreq)

mlp = Chain(
    Dense(18, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, nfreq),
    SinWaveEmbedder(latent_dim, nfreq),
    c -> 2.0f0 * sigmoid.(c))

design_encoder = DesignEncoder(env.design_space, mlp, env.integration_steps)
F = SinusoidalSource(latent_dim, nfreq, 1000.0f0)
dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = Integrator(runge_kutta, dyn, env.dt)
model = gpu(AcousticEnergyModel(wave_encoder, design_encoder, F, iter))

opt_state = Optimisers.setup(Optimisers.Adam(1e-3), model)

for i in 1:10
    loss, back = Flux.pullback(model) do m
        z = m(s, a, t)
        energy = latent_energy(z, dx)
        Flux.mse(energy, y)
    end

    @time gs = back(one(loss))[1]
    opt_state, model = Optimisers.update(opt_state, model, gs)
    println("Loss: $loss")
end

z = model(s, a, t)
energy = cpu(latent_energy(z, dx))
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
lines!(ax1, tspan, true_energy[:, 1, 1])
lines!(ax1, tspan, true_energy[:, 2, 1])
lines!(ax1, tspan, true_energy[:, 3, 1])
lines!(ax2, tspan, energy[:, 1, 1])
lines!(ax2, tspan, energy[:, 2, 1])
lines!(ax2, tspan, energy[:, 3, 1])
save("latent_energy_post_optimization.png", fig)

# z = cpu(z)
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)
# record(fig, "latent_post_optimization.mp4", axes(t, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
#     lines!(ax, latent_dim.x, z[:, 3, 1, i], color = :orange)
# end