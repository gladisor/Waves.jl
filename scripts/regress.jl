using Waves, Flux, Optimisers
using CairoMakie
using BSON
Flux.CUDA.allowscalar(false)

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")

horizon = 10
batchsize = 1
h_size = 256
nfreq = 500
elements = 1024
latent_dim = OneDim(100.0f0, elements)
dx = get_dx(latent_dim)

data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)

wave_encoder = gpu(Waves.build_wave_encoder(;latent_dim, nfreq, h_size, c0 = env.iter.dynamics.c0))

mlp = Chain(
    Dense(18, h_size, leakyrelu), 
    Dense(h_size, h_size, leakyrelu), 
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu), 
    Dense(h_size, nfreq),
    Waves.SinWaveEmbedder(latent_dim, nfreq),
    c -> 2.0f0 * sigmoid.(c))

design_encoder = Waves.DesignEncoder(env.design_space, mlp, env.integration_steps)
F = Waves.SinusoidalSource(latent_dim, nfreq, env.source.freq)
dyn = Waves.AcousticDynamics(latent_dim, env.iter.dynamics.c0, 10.0f0, 10000.0f0)
iter = Integrator(runge_kutta, dyn, env.dt)

s, a, t, y = gpu(Flux.batch.(first(data)))

iter = gpu(iter)
design_encoder = gpu(design_encoder)
F = gpu(F)
z0 = gpu(zeros(Float32, elements, 4, 1))
C = design_encoder(s, a, t)
target = gpu(build_normal(build_grid(latent_dim), [0.0f0], [3.0f0], [5.0f0]))

theta = [C, F]
opt_state = Optimisers.setup(Optimisers.Adam(1f-2), theta)

for i in 1:10
    loss, back = Flux.pullback(theta) do _theta
        z = iter(z0, t, _theta)
        return Flux.mse(z[:, 1, 1, end], target)
    end

    gs = back(one(loss))[1]
    println(loss)
    opt_state, theta = Optimisers.update(opt_state, theta, gs)
end

z = cpu(iter(z0, t, theta))

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "latent.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
    lines!(ax, latent_dim.x, cpu(target), color = :red)
end



### For training model
# model = gpu(AcousticEnergyModel(wave_encoder, design_encoder, F, iter, dx))
# opt_state = Optimisers.setup(Optimisers.Adam(1f-4), model)

# for epoch in 1:10
#     for (i, batch) in enumerate(data)
#         s, a, t, y = gpu(Flux.batch.(batch))
#         loss, back = Flux.pullback(m -> Flux.mse(m(s, a, t), y), model)

#         println("Iteration: $i, Loss: $loss")
#         @time gs = back(one(loss))[1]
#         opt_state, model = Optimisers.update(opt_state, model, gs)
#     end
# end

# s, a, t, y = gpu(Flux.batch.(first(data)))
# evaluate(model, s, a, t, y)



# true_energy = cpu(y[:, :, 1])
# energy = cpu(model(s, a, t)[:, :, 1])
# tspan = cpu(t[:, 1])

# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "Latent Energy", xlabel = "Time (s)")
# ax2 = Axis(fig[1, 2], title = "Real Energy", xlabel = "Time (s)")
# lines!(ax1, tspan, energy[:, 1], label = "Total")
# lines!(ax1, tspan, energy[:, 2], label = "Incident")
# lines!(ax1, tspan, energy[:, 3], label = "Scattered")
# lines!(ax2, tspan, true_energy[:, 1], label = "Total")
# lines!(ax2, tspan, true_energy[:, 2], label = "Incident")
# lines!(ax2, tspan, true_energy[:, 3], label = "Scattered")
# # save("$(typeof(model.F))_nfreq=$(nfreq)_energy.png", fig)
# save("y1.png", fig)

# true_energy = cpu(y[:, :, 2])
# energy = cpu(model(s, a, t)[:, :, 2])
# tspan = cpu(t[:, 2])

# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "Latent Energy", xlabel = "Time (s)")
# ax2 = Axis(fig[1, 2], title = "Real Energy", xlabel = "Time (s)")
# lines!(ax1, tspan, energy[:, 1], label = "Total")
# lines!(ax1, tspan, energy[:, 2], label = "Incident")
# lines!(ax1, tspan, energy[:, 3], label = "Scattered")
# lines!(ax2, tspan, true_energy[:, 1], label = "Total")
# lines!(ax2, tspan, true_energy[:, 2], label = "Incident")
# lines!(ax2, tspan, true_energy[:, 3], label = "Scattered")
# # save("$(typeof(model.F))_nfreq=$(nfreq)_energy.png", fig)
# save("y2.png", fig)








# z = cpu(generate_latent_solution(model, s, a, t))

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)

# record(fig, "latent1.mp4", axes(t, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
# end

# record(fig, "latent2.mp4", axes(t, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, z[:, 1, 2, i], color = :blue)
# end


# fig = Figure()
# ax = Axis(fig[1,1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -2.0f0, 2.0f0)

# record(fig, "F.mp4", axes(t, 1)) do i
#     empty!(ax)
#     lines!(ax, latent_dim.x, cpu(model.F(t[i, :])[:, 1]), color = :blue)
# end