using Waves, Flux, CairoMakie, BSON
using Optimisers

Flux.device!(0)
Flux.CUDA.allowscalar(false)

dt = 1f-5
N = 300
nfreq = 50

grid_size = 15.0f0
elements = 1024

latent_dim = OneDim(grid_size, elements)
dx = get_dx(latent_dim)

dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, dt))

target = gpu(build_normal(latent_dim.x, [0.0f0], [0.3f0], [1.0f0]))

freq_coefs = gpu(randn(Float32, nfreq, 4, 1)) * 0.01f0
emb = gpu(SinWaveEmbedder(latent_dim, nfreq))

t = gpu(build_tspan(0.0f0, dt, N))[:, :]

C = LinearInterpolation(t[[1, end], :], gpu(ones(Float32, elements, 2, 1)))
F = Source(zeros(Float32, elements), 1.0f0)
θ = gpu([C, F, dyn.pml / maximum(dyn.pml)])

pre_opt_z = cpu(iter(emb(freq_coefs), t, θ))

opt_state = Optimisers.setup(Optimisers.Adam(5e-2), freq_coefs)

for i in 1:10
    loss, back = Flux.pullback(freq_coefs) do _freq_coefs
        z = iter(emb(_freq_coefs), t, θ)
        return Flux.mse(z[:, 1, 1, end], target) + Flux.norm(_freq_coefs) * 0.005f0
    end

    println(loss)
    gs = back(one(loss))[1]
    opt_state, freq_coefs = Optimisers.update(opt_state, freq_coefs, gs)
end

z0 = emb(freq_coefs)
z = cpu(iter(z0, t, θ))

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0, 2.0)
record(fig, "vid.mp4", axes(z, 4)) do i 
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
end

fig = Figure()
ax1 = Axis(fig[1, 1], title = "Before Optimization", xlabel = "Space (m)", ylabel = "Time (s)")
ax2 = Axis(fig[1, 2], title = "After Optimization", xlabel = "Space (m)", ylabel = "Time (s)")
heatmap!(ax1, latent_dim.x, cpu(vec(t)), pre_opt_z[:, 1, 1, :], colormap = :ice)
heatmap!(ax2, latent_dim.x, cpu(vec(t)), z[:, 1, 1, :], colormap = :ice)
save("wave.png", fig)