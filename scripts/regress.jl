using Waves, Flux, Optimisers
using CairoMakie
using BSON
Flux.CUDA.allowscalar(false)
include("model.jl")

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")

horizon = 3
batchsize = 1
h_size = 256
nfreq = 500

latent_dim = OneDim(100.0f0, 1024)
dx = get_dx(latent_dim)

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))
tspan = cpu(vec(t))

wave_encoder = gpu(build_wave_encoder(;latent_dim, nfreq, h_size, c0 = env.iter.dynamics.c0))

mlp = Chain(
    Dense(18, h_size, leakyrelu), 
    Dense(h_size, h_size, leakyrelu), 
    Dense(h_size, nfreq),
    SinWaveEmbedder(latent_dim, nfreq),
    c -> 2.0f0 * sigmoid.(c))

design_encoder = DesignEncoder(env.design_space, mlp, env.integration_steps)
F = SinusoidalSource(latent_dim, nfreq, env.source.freq)
# F = GaussianSource(latent_dim, [0.0f0], [log(0.3f0)], [1.0f0], env.source.freq)

dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, 5.0f0, 10000.0f0)
iter = Integrator(runge_kutta, dyn, env.dt)
model = gpu(AcousticEnergyModel(wave_encoder, design_encoder, F, iter, dx))

opt_state = Optimisers.setup(Optimisers.Adam(1f-4), model)

for i in 1:50
    loss, back = Flux.pullback(model) do m
        return Flux.mse(m(s, a, t), y)
    end

    println("Iteration: $i, Loss: $loss")
    @time gs = back(one(loss))[1]
    opt_state, model = Optimisers.update(opt_state, model, gs)
end

true_energy = cpu(y)
energy = cpu(model(s, a, t))
fig = Figure()
ax1 = Axis(fig[1, 1], title = "Latent Energy", xlabel = "Time (s)")
ax2 = Axis(fig[1, 2], title = "Real Energy", xlabel = "Time (s)")
lines!(ax1, tspan, energy[:, 1], label = "Total")
lines!(ax1, tspan, energy[:, 2], label = "Incident")
lines!(ax1, tspan, energy[:, 3], label = "Scattered")
lines!(ax2, tspan, true_energy[:, 1], label = "Total")
lines!(ax2, tspan, true_energy[:, 2], label = "Incident")
lines!(ax2, tspan, true_energy[:, 3], label = "Scattered")
save("$(typeof(model.F))_nfreq=$(nfreq)_energy.png", fig)

z = cpu(generate_latent_solution(model, s, a, t))

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)

record(fig, "latent.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
end

fig = Figure()
ax = Axis(fig[1,1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)

record(fig, "F.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, cpu(vec(model.F(t[i, :]))), color = :blue)
end