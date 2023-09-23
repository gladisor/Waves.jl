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
# F = gpu(SinusoidalSource(latent_dim, nfreq, env.source.freq))
F = gpu(GaussianSource(latent_dim, [0.0f0], [0.3f0], [0.0f0], env.source.freq))
# f = gpu(zeros(Float32, size(latent_dim)..., batchsize))
θ = [C, F]

dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

c0 = env.iter.dynamics.c0
emb = gpu(Chain(SinWaveEmbedder(latent_dim, nfreq), x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0)))
freq_coefs = gpu(randn(Float32, nfreq, 4, 1)) * 0.1f0

z0 = emb(freq_coefs)# * 0.0f0
#z0[:, 1, 1] .= gpu(build_normal(latent_dim.x, [0.0f0], [1.0f0], [1.0f0]))
target = gpu(build_normal(latent_dim.x, [5.0f0], [0.3f0], [1.0f0]))

opt_state = Optimisers.setup(Optimisers.Adam(1e-1), freq_coefs)

for i in 1:10
    loss, back = Flux.pullback(freq_coefs) do _freq_coefs
        # z = iter(_z0, t, θ)
        z = iter(emb(_freq_coefs), t, θ)
        return Flux.mse(z[:, 1, 1, end], target) + Flux.norm(_freq_coefs) * 0.005f0
    end

    println(loss)
    gs = back(one(loss))[1]
    opt_state, freq_coefs = Optimisers.update(opt_state, freq_coefs, gs)
end

z = iter(emb(freq_coefs), t, θ)
z = cpu(z)

fig = Figure()
ax = Axis(fig[1, 1], title = "Wave Simulation Optimization", xlabel = "Space (m)", ylabel = "Displacement (m)")
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)

record(fig, "latent_freq_opt.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, cpu(target), color = :red)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
end




# @time render(latent_dim, cpu(z[:, :, 1, :]), tspan, path = "latent.mp4")

# x = [freq_coefs, θ]
# z0 = emb(x[1])
# @time z = iter(emb(x[1]), t, x[2])
# opt_state = Optimisers.setup(Optimisers.Descent(1e-3), z0)

# for i in 1:10
#     loss, back = Flux.pullback(x) do _x
#         z = iter(emb(_x[1]), t, _x[2])
#         sum(z[:, 1, 1, end] .^ 2) * dx
#     end

#     println(loss)
#     gs = back(one(loss))[1]
#     opt_state, x = Optimisers.update(opt_state, x, gs)
# end















# z = iter(emb(x[1]), t, x[2])
# energy = cpu(latent_energy(z, dx))
# plot_energy(tspan, energy[:, :, 1], path = "pre_opt_latent_energy.png")
# plot_energy(tspan, cpu(y[:, :, 1]), path = "true_energy.png")

# z = iter(z0, t, θ)
# @time render(latent_dim, cpu(z[:, :, 1, :]), tspan, path = "latent.mp4")

# opt_state = Optimisers.setup(Optimisers.Descent(1e-3), x)

# for i in 1:20
#     loss, back = Flux.pullback(x) do _x
#         z = iter(emb(_x[1]), t, _x[2])
#         e = latent_energy(z, dx)
#         return Flux.mse(e, y)
#     end
#     gs = back(one(loss))[1]

#     opt_state, x = Optimisers.update(opt_state, x, gs)
#     println(loss)
# end

# z = iter(emb(x[1]), t, x[2])
# energy = cpu(latent_energy(z, dx))
# plot_energy(tspan, energy[:, :, 1], path = "post_opt_latent_energy.png")