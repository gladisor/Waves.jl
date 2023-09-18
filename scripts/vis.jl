using Waves
using Flux
using CairoMakie
using BSON

include("model.jl")

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")
length(ep)

horizon = 20
data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = 1, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))

latent_dim = OneDim(15.0f0, 700)
wave_encoder = build_wave_encoder(;latent_dim)

h_size = 256
nfreq = 50
mlp = Chain(
    Dense(18, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, h_size, leakyrelu),
    Dense(h_size, nfreq),
    SinWaveEmbedder(latent_dim, nfreq),
    c -> 2.0f0 * sigmoid.(c)
)

design_encoder = gpu(DesignEncoder(env.design_space, mlp, env.integration_steps))
C = design_encoder(s, a, t)
# F = gpu(GaussianSource(latent_dim, [0.0f0], [0.3f0], [1.0f0], 1000.0f0))
F = gpu(SinusoidalSource(latent_dim, nfreq, 1000.0f0))

dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))
z0 = gpu(zeros(Float32, size(latent_dim)..., 4, 1))
z = cpu(iter(z0, t, [C, F]))
tspan = vec(cpu(t))




dx = get_dx(latent_dim)
tot = z[:, 1, 1, :]
inc = z[:, 3, 1, :]
sc = tot .- inc
tot_energy = vec(sum(tot .^ 2, dims = 1)) * dx
inc_energy = vec(sum(inc .^ 2, dims = 1)) * dx
sc_energy = vec(sum(sc .^ 2, dims = 1)) * dx

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, tspan, tot_energy)
lines!(ax, tspan, inc_energy)
lines!(ax, tspan, sc_energy)
save("latent_energy.png", fig)





fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)
record(fig, "latent.mp4", axes(t, 1)) do i
    empty!(ax)
    # lines!(ax, latent_dim.x, tot[:, i], color = :blue)
    # lines!(ax, latent_dim.x, inc[:, i], color = :orange)
    lines!(ax, latent_dim.x, sc[:, i], color = :red)
end
