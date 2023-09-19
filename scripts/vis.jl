using Waves
using Flux
using Optimisers
using CairoMakie
using BSON

include("model.jl")

function latent_energy(z::AbstractArray{Float32, 4}, dx::Float32)
    tot = z[:, 1, :, :]
    inc = z[:, 3, :, :]
    sc = tot .- inc

    tot_energy = sum(tot .^ 2, dims = 1) * dx
    inc_energy = sum(inc .^ 2, dims = 1) * dx
    sc_energy =  sum(sc  .^ 2, dims = 1) * dx
    return permutedims(vcat(tot_energy, inc_energy, sc_energy), (3, 1, 2))
end

env = BSON.load("env.bson")[:env]
ep = Episode(path = "episode.bson")
length(ep)

horizon = 2
batchsize = 1
data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = batchsize, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))
tspan = vec(cpu(t))
true_energy = cpu(y)


latent_dim = OneDim(100.0f0, 700)
dx = get_dx(latent_dim)
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
θ = [C, F]

dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

emb = gpu(SinWaveEmbedder(latent_dim, nfreq))
freq_coefs = gpu(randn(Float32, nfreq, 4, 1) ./ nfreq)
weights = gpu(reshape([1.0f0, 1.0f0 / WATER, 1.0f0, 1.0f0 / WATER], (1, 4, 1)))






z0 = emb(freq_coefs) .* weights
z = iter(z0, t, θ)
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
save("latent_energy_pre_optimization.png", fig)

z = cpu(z)
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -2.0f0, 2.0f0)
record(fig, "latent_pre_optimization.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
    lines!(ax, latent_dim.x, z[:, 3, 1, i], color = :orange)
end










opt_state = Optimisers.setup(Optimisers.Adam(1e-2), freq_coefs)
θ = [C, F]

for i in 1:50
    loss, back = Flux.pullback(freq_coefs) do _freq_coefs
        z0 = emb(_freq_coefs) .* weights
        y_hat = latent_energy(iter(z0, t, θ), dx)
        return Flux.mse(y_hat, y)
    end

    gs = back(one(loss))[1]
    opt_state, freq_coefs = Optimisers.update(opt_state, freq_coefs, gs)
    println("Loss: $loss")
end










z0 = emb(freq_coefs) .* weights
z = iter(z0, t, θ)
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

z = cpu(z)
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)
record(fig, "latent_post_optimization.mp4", axes(t, 1)) do i
    empty!(ax)
    lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
    lines!(ax, latent_dim.x, z[:, 3, 1, i], color = :orange)
end