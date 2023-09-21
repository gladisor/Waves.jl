using Waves, Flux, Optimisers
Flux.CUDA.allowscalar(false)
using CairoMakie
using BSON
include("model.jl")

function compute_gs_ad(emb::Chain, iter::Integrator, freq_coefs::AbstractArray{Float32, 3}, t::AbstractMatrix{Float32}, θ, dx::Float32, y::AbstractArray{Float32, 3})
    z0, back_freq_coefs = Flux.pullback(_freq_coefs -> emb(_freq_coefs), freq_coefs)

    loss, back = Flux.pullback(z0) do _z0
        z = iter(_z0, t, θ)
        e = latent_energy(z, dx)
        return Flux.mse(e, y)
    end

    gs_AD = back(one(loss))[1]
    return loss, back_freq_coefs(gs_AD)[1]
end

function compute_gs_as(emb::Chain, iter::Integrator, freq_coefs::AbstractArray{Float32, 3}, t::AbstractMatrix{Float32}, θ, dx::Float32, y::AbstractArray{Float32, 3})
    z0, back_freq_coefs = Flux.pullback(_freq_coefs -> emb(_freq_coefs), freq_coefs)
    z = iter(z0, t, θ)

    loss, back = Flux.pullback(z) do _z
        e = latent_energy(_z, dx)
        return Flux.mse(e, y)
    end

    adj = back(one(loss))[1]
    gs_AS = adjoint_sensitivity(iter, z, t, θ, adj)
    return loss, back_freq_coefs(gs_AS)[1]
end

function optimise_freq_coefs(emb, iter, freq_coefs, t, θ, dx, y, gs_fxn::Function, lr::Float32)
    opt_state = Optimisers.setup(Optimisers.Descent(lr), freq_coefs)

    for i in 1:50
        loss, gs = gs_fxn(emb, iter, freq_coefs, t, θ, dx, y)
        opt_state, freq_coefs = Optimisers.update(opt_state, freq_coefs, gs)
        println(loss)
    end

    return freq_coefs
end

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
F = gpu(SinusoidalSource(latent_dim, nfreq, env.source.freq))
θ = [C, F]

dyn = AcousticDynamics(latent_dim, env.iter.dynamics.c0, 5.0f0, 10000.0f0)
iter = gpu(Integrator(runge_kutta, dyn, env.dt))

c0 = env.iter.dynamics.c0
emb = gpu(Chain(
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0)))
        
freq_coefs = gpu(randn(Float32, nfreq, 4, 1))




z = iter(emb(freq_coefs), t, θ)
energy = cpu(latent_energy(z, dx))
plot_energy(tspan, energy[:, :, 1], path = "pre_opt_latent_energy.png")
plot_energy(tspan, cpu(y[:, :, 1]), path = "true_energy.png")

# lr = 1f-3 # AS
lr = 1f-2
# lr = 1f-4 # AD
freq_coefs = optimise_freq_coefs(emb, iter, freq_coefs, t, θ, dx, y, compute_gs_as, lr)
# freq_coefs = optimise_freq_coefs(emb, iter, freq_coefs, t, θ, dx, y, compute_gs_ad, lr)

z = iter(emb(freq_coefs), t, θ)
energy = cpu(latent_energy(z, dx))
plot_energy(tspan, energy[:, :, 1], path = "post_opt_latent_energy.png")









# gs_AD = cpu(compute_gs_ad(emb, iter, freq_coefs, t, θ, dx, y))
# gs_AS = cpu(compute_gs_as(emb, iter, freq_coefs, t, θ, dx, y))


# fig = Figure()
# ax1 = Axis(fig[1, 1], xlabel = "Space (m)", ylabel = "Autodiff Gradient", title = "Gradient of energy difference with respect to initial total displacement")
# ax2 = Axis(fig[1, 1], yaxisposition = :right, ylabel = "Adjoint Sensitivity Gradient")
# lines!(ax1, latent_dim.x, gs_AD[:, 1, 1], color = :blue)
# lines!(ax2, latent_dim.x, gs_AS[:, 1, 1], color = :orange)
# save("gs.png", fig)
# render(latent_dim, cpu(z[:, :, 1, :]), tspan, path = "latent.mp4")