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
dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

c = gpu(ones(Float32, size(latent_dim)..., batchsize))
f = gpu(zeros(Float32, size(latent_dim)..., batchsize))
C = t -> c
# F = t -> f
F = gpu(SinusoidalSource(latent_dim, nfreq, env.source.freq))
# F = gpu(GaussianSource(latent_dim, [0.0f0], [0.3f0], [1.0f0], env.source.freq))
θ = [C, F]

wave_encoder = gpu(build_wave_encoder(;latent_dim, nfreq))

opt_state = Optimisers.setup(Optimisers.Adam(1e-3), wave_encoder)

z0_pre_opt = cpu(wave_encoder(s))
z_pre_opt = iter(gpu(z0_pre_opt), t, θ)
energy_pre_opt = cpu(latent_energy(z_pre_opt, dx))

true_energy = cpu(y)
tspan = vec(cpu(t))

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
lines!(ax1, tspan, true_energy[:, 1, 1])
lines!(ax1, tspan, true_energy[:, 2, 1])
lines!(ax1, tspan, true_energy[:, 3, 1])
lines!(ax2, tspan, energy_pre_opt[:, 1, 1])
lines!(ax2, tspan, energy_pre_opt[:, 2, 1])
lines!(ax2, tspan, energy_pre_opt[:, 3, 1])
save("latent_energy_pre_optimization.png", fig)

z = cpu(z_pre_opt)
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)

record(fig, "latent.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
    lines!(ax, latent_dim.x, cpu(vec(F(t[i, :]))), color = :red)
end

# for i in 1:10
loss, back = Flux.pullback(wave_encoder) do we
    z = iter(we(s), t, θ)
    energy = latent_energy(z, dx)
    Flux.mse(energy, y)
end

# println("Iteration: $i, Loss: $loss")
@time gs = back(one(loss))[1]
# opt_state, wave_encoder = Optimisers.update(opt_state, wave_encoder, gs)
# end

# z0_post_opt = cpu(wave_encoder(s))

# z_post_opt = iter(gpu(z0_post_opt), t, θ)
# energy_post_opt = cpu(latent_energy(z_post_opt, dx))

# z = cpu(z_post_opt)
# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)

# record(fig, "latent.mp4", axes(t, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
#     lines!(ax, latent_dim.x, cpu(vec(F(t[i, :]))), color = :red)
# end

# true_energy = cpu(y)
# tspan = vec(cpu(t))

# fig = Figure()
# ax1 = Axis(fig[1, 1])
# ax2 = Axis(fig[1, 2])
# lines!(ax1, tspan, true_energy[:, 1, 1])
# lines!(ax1, tspan, true_energy[:, 2, 1])
# lines!(ax1, tspan, true_energy[:, 3, 1])
# lines!(ax2, tspan, energy_post_opt[:, 1, 1])
# lines!(ax2, tspan, energy_post_opt[:, 2, 1])
# lines!(ax2, tspan, energy_post_opt[:, 3, 1])
# save("latent_energy_post_optimization.png", fig)