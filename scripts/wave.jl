using Waves
using Flux
Flux.CUDA.allowscalar(false)
using Optimisers
# using Flux.ChainRulesCore: Tangent, ZeroTangent
using CairoMakie
using ReinforcementLearning

function energy_gradient_ad(iter::Integrator, wave::AbstractArray, t::AbstractMatrix, θ, y::AbstractArray)

    loss, back = Flux.pullback(wave, θ) do _wave, _θ
        z = iter(_wave, t, _θ)

        u_tot = z[:, 1, 1, :]
        u_inc = z[:, 3, 1, :]
        u_sc = u_tot .- u_inc
        tot_energy = vec(sum(u_tot .^ 2, dims = 1))
        inc_energy = vec(sum(u_inc .^ 2, dims = 1))
        sc_energy  = vec(sum(u_sc  .^ 2, dims = 1))

        y_hat = hcat(tot_energy, inc_energy, sc_energy)
        return Flux.mse(y_hat, y[:, :, 1])
    end

    return back(one(loss))
end

function optimise!(iter::Integrator, wave::AbstractArray, t::AbstractMatrix, θ, y::AbstractArray)

    opt_state = Optimisers.setup(Optimisers.Adam(1e-3), θ)

    for i in 1:10
        gs = energy_gradient_ad(iter, wave, t, θ, y)
        println("Step: $i, Loss: $loss")
        opt_state, θ = Optimisers.update(opt_state, θ, gs[2])
    end
end

# """
# adjoint_sensitivity method specifically for differentiating a batchwise OneDim simulation.

# u: (finite elements x fields x batch x time)
# t: (time x batch)
# adj: same as solution (u)
# """
# function adjoint_sensitivity(iter::Integrator, z::AbstractArray{Float32, 4}, t::AbstractMatrix{Float32}, θ, ∂L_∂z::AbstractArray{Float32, 4})
#     ∂L_∂z₀ = ∂L_∂z[:, :, :, 1] * 0.0f0 ## loss accumulator
#     # ∂L_∂θ = ZeroTangent()

#     for i in axes(z, 4)
#         zᵢ = z[:, :, :, i]
#         tᵢ = t[i, :]
#         aᵢ = ∂L_∂z[:, :, :, i]

#         _, back = Flux.pullback(zᵢ, θ) do _zᵢ, _θ
#             return iter.integration_function(iter.dynamics, _zᵢ, tᵢ, _θ, iter.dt)
#         end
        
#         ∂aᵢ_∂tᵢ, ∂aᵢ_∂θ = back(∂L_∂z₀)
#         ∂L_∂z₀ .+= aᵢ .+ ∂aᵢ_∂tᵢ
#     end

#     return ∂L_∂z₀
# end

function build_wave_encoder(;
        k::Tuple{Int, Int} = (3, 3),
        in_channels::Int = 3,
        activation::Function = leakyrelu,
        h_size::Int = 256,
        nfields::Int = 4,
        nfreq::Int = 50,
        c0::Float32 = WATER,
        latent_dim::OneDim)

    return Chain(
        Waves.TotalWaveInput(),
        Waves.ResidualBlock(k, in_channels, 32, activation),
        Waves.ResidualBlock(k, 32, 64, activation),
        Waves.ResidualBlock(k, 64, h_size, activation),
        GlobalMaxPool(),
        Flux.flatten,
        Dense(h_size, nfields * nfreq, tanh),
        b -> reshape(b, nfreq, nfields, :),
        Waves.SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0))
end

struct DesignEncoder
    design_space::DesignSpace
    layers::Chain
    integration_steps::Int
end

Flux.@functor DesignEncoder
Flux.trainable(de::DesignEncoder) = (;de.layers)

function (de::DesignEncoder)(d1::Vector{<: AbstractDesign}, a::Vector{<: AbstractDesign})
    d2 = de.design_space.(d1, a)
    return (d2, d2)
end

function (de::DesignEncoder)(s::Vector{WaveEnvState}, a::Matrix{<:AbstractDesign}, t::AbstractMatrix{Float32})
    t_ = t[1:de.integration_steps:end, :]
    d = [si.design for si in s]
    recur = Flux.Recur(de, d)
    design_sequences = hcat(d, [recur(a[i, :]) for i in axes(a, 1)]...)
    x = Waves.normalize.(design_sequences, [de.design_space])
    x_batch = cat([hcat(x[i, :]...) for i in axes(x, 1)]..., dims = 3)
    y = de.layers(x_batch)
    return LinearInterpolation(t_, y)
end

dim = TwoDim(15.0f0, 700)
n = 10
μ = zeros(Float32, n, 2)
μ[:, 1] .= -10.0f0
μ[:, 2] .= range(-2.0f0, 2.0f0, n)

σ = ones(Float32, n) * 0.3f0
a = ones(Float32, n) * 0.3f0
pulse = build_normal(build_grid(dim), μ, σ, a)
source = Source(pulse, 1000.0f0)

env = gpu(WaveEnv(dim; 
    design_space = Waves.build_triple_ring_design_space(),
    source = source,
    integration_steps = 100,
    actions = 10))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env, path = "vid.mp4")
ep = generate_episode!(policy, env)

# horizon = 5
# data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = 2, shuffle = true, partial = false)
# s, a, t, y = gpu(Flux.batch.(first(data)))
# latent_dim = OneDim(15.0f0, 700)

# nfreq = 50
# in_size = 18
# h_size = 256
# activation = leakyrelu

# dyn = AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0)
# iter = gpu(Integrator(runge_kutta, dyn, env.dt))
# wave_encoder = gpu(build_wave_encoder(;latent_dim, nfreq, h_size))
# wave = wave_encoder(s)

# mlp = Chain(
#     Dense(in_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, h_size, activation),
#     Dense(h_size, nfreq, tanh),
#     Waves.SinWaveEmbedder(latent_dim, nfreq),
#     sigmoid
#     )

# de = gpu(DesignEncoder(env.design_space, mlp, env.integration_steps))

# C = de(s, a, t)
# F = gpu(Waves.SinusoidalSource(latent_dim, nfreq, 1000.0f0))
# θ = [C, F]

# c = cpu(hcat([C(t[i, :]) for i in axes(t, 1)]...))














# fig = Figure()
# ax = Axis(fig[1, 1])
# heatmap!(ax, latent_dim.x, cpu(t[:, 1]), c, colormap = :ice)
# save("c.png", fig)

# ### Plotting Gradients
# dwave = gs[1]
# fig = Figure(resolution = (1920, 1080), fontsize = 50)
# ax1 = Axis(fig[1, 1], title = L"\frac{\partial L}{\partial u_0}", xlabel = "Space (m)")
# ax2 = Axis(fig[1, 2], title = L"\frac{\partial L}{\partial v_0}", xlabel = "Space (m)")

# lines!(ax1, latent_dim.x, dwave[:, 1, 1], label = "Total", linewidth = 3)
# lines!(ax1, latent_dim.x, dwave[:, 3, 1], label = "Incident", linewidth = 3)

# lines!(ax2, latent_dim.x, dwave[:, 2, 1], label = "Total", linewidth = 3)
# lines!(ax2, latent_dim.x, dwave[:, 4, 1], label = "Incident", linewidth = 3)
# axislegend(ax1, position = :lt)
# axislegend(ax2, position = :lt)
# save("dwave_gaussian.png", fig)

# ### rendering simulation
# z = cpu(z)






# @time z = cpu(iter(wave, t, θ))
# u_tot = z[:, 1, 1, :]
# u_inc = z[:, 3, 1, :]
# u_sc = u_tot .- u_inc

fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, latent_dim.x[1], latent_dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)

record(fig, "latent_gaussian.mp4", axes(t, 1)) do i
    empty!(ax)

    c = cpu(C(t[i, :]))
    # f = cpu(F(t[i, :]))
    # lines!(ax, latent_dim.x, u_tot[:, i], color = :blue)
    # lines!(ax, latent_dim.x, u_inc[:, i], color = :green)
    # lines!(ax, latent_dim.x, u_sc[:, i], color = :red)
    # lines!(ax, latent_dim.x, c[:, 1], color = :grey)
    # lines!(ax, latent_dim.x, f[:, 1], color = :blue)
    lines!(ax, latent_dim.x, c[:, 1], color = :blue)
end

# ### Plotting energy
# tot_energy = vec(sum(u_tot .^ 2, dims = 1))
# inc_energy = vec(sum(u_inc .^ 2, dims = 1))
# sc_energy  = vec(sum(u_sc  .^ 2, dims = 1))

# fig = Figure()
# ax1 = Axis(fig[1, 1], title = "Latent Energy over Trajectory", xlabel = "Time (s)", ylabel = "Energy")
# tspan = cpu(t[:, 1])
# lines!(ax1, tspan, tot_energy, color = :blue, label = "Total")
# lines!(ax1, tspan, inc_energy, color = :green, label = "Incident")
# lines!(ax1, tspan, sc_energy, color = :red, label = "Scattered")
# axislegend(ax1, position = :lt)

# y = cpu(y)
# ax2 = Axis(fig[1, 2])
# lines!(ax2, tspan, y[:, 1, 1], color = :blue, label = "Total")
# lines!(ax2, tspan, y[:, 2, 1], color = :green, label = "Incident")
# lines!(ax2, tspan, y[:, 3, 1], color = :red, label = "Scattered")
# axislegend(ax2, position = :lt)
# save("latent_energy_gaussian.png", fig)








# # # """
# # # Testing with known function interp
# # # """
# # # y = sin.(1000.0f0 .* t_)
# # # y_true = sin.(1000.0f0 .* t)
# # # # interp = gpu(PolynomialInterpolation(t′, Flux.unsqueeze(y, 1)))
# # # # y_pred = vcat([interp(t[i, :]) for i in axes(t, 1)]...)
# # # y_pred = vcat([linear_interp(t_, Flux.unsqueeze(y, 1), t[i, :]) for i in axes(t, 1)]...)

# # # fig = Figure()
# # # ax = Axis(fig[1, 1])
# # # scatter!(ax, cpu(t_[:, 1]), cpu(y[:, 1]))
# # # lines!(ax, cpu(t[:, 1]), cpu(y_pred[:, 1]), color = :green)
# # # lines!(ax, cpu(t[:, 1]), cpu(y_true[:, 1]), color = :orange)
# # # save("interp.png", fig)

# # # """
# # # Testing AD vs AS
# # # """
# # # # z, back = Flux.pullback(wave) do _wave
# # # #     z = iter(_wave, t, θ)
# # # # end

# # # # adj = z .^ 0.0f0

# # # # @time gs_AD = cpu(back(adj)[1])
# # # # @time gs_AS = cpu(adjoint_sensitivity(iter, z, t, θ, adj))

# # # # fig = Figure()
# # # # ax1 = Axis(fig[1, 1])
# # # # ax2 = Axis(fig[1, 1])
# # # # ax2.yaxisposition = :right
# # # # ax3 = Axis(fig[1, 2])
# # # # ax4 = Axis(fig[1, 2])
# # # # ax4.yaxisposition = :right

# # # # lines!(ax1, gs_AD[:, 1, 1], color = :blue)
# # # # lines!(ax2, gs_AS[:, 1, 1], color = :orange)
# # # # lines!(ax3, gs_AD[:, 2, 1], color = :blue)
# # # # lines!(ax4, gs_AS[:, 2, 1], color = :orange)
# # # # save("gs.png", fig)
