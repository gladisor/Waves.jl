using Waves
using Flux
Flux.CUDA.allowscalar(false)
# using Flux.ChainRulesCore: Tangent, ZeroTangent
using CairoMakie
using ReinforcementLearning
# using LinearAlgebra

struct Episode{S, Y}
    s::Vector{S}
    a::Vector{<: AbstractDesign}
    t::Vector{Vector{Float32}}
    y::Vector{Y}
end

Base.length(ep::Episode) = length(ep.s)

function generate_episode!(policy::AbstractPolicy, env::WaveEnv)
    s = WaveEnvState[]
    a = AbstractDesign[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]

    reset!(env)
    while !is_terminated(env)
        push!(s, state(env))
        action = policy(env)
        push!(a, cpu(action))
        push!(t, build_tspan(env))
        env(action)
        push!(y, cpu(env.signal))
        println(env.time_step)
    end

    return Episode(s, a, t, y)
end

function prepare_data(ep::Episode{S, Matrix{Float32}}, horizon::Int) where S
    s = S[]
    a = Vector{<: AbstractDesign}[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]
    
    n = horizon - 1
    for i in 1:(length(ep) - n)
        boundary = i + n
        push!(s, ep.s[i])
        push!(a, ep.a[i:boundary])
        push!(t, flatten_repeated_last_dim(hcat(ep.t[i:boundary]...)))

        signal = cat(ep.y[i:boundary]..., dims = 3)
        signal = permutedims(flatten_repeated_last_dim(permutedims(signal, (2, 1, 3))))
        push!(y, signal)
    end

    return s, a, t, y
end

function prepare_data(eps::Vector{Episode{S, Y}}, horizon::Int) where {S, Y}
    return vcat.(prepare_data.(eps, horizon)...)
end

"""
mask:   (sequence x sequence)
x:      (sequence x batch)
y:      (features x sequence x batch)
"""
struct PolynomialInterpolation
    mask::AbstractMatrix
    x::AbstractMatrix
    y::AbstractArray
end

Flux.@functor PolynomialInterpolation
Flux.trainable(interp::PolynomialInterpolation) = (;interp.y)

function PolynomialInterpolation(x::AbstractArray, y::AbstractArray)
    mask = I(size(x, 1))
    return PolynomialInterpolation(mask, x, y)
end

function (interp::PolynomialInterpolation)(t::AbstractVector{Float32})
    scale = Flux.unsqueeze(maximum(abs.(interp.x), dims = 1), 1)
    n = interp.mask .+ (.!interp.mask) .* Flux.unsqueeze(interp.x .- permutedims(t), 2)
    numer = Flux.prod(n ./ scale .+ 1f-5, dims = 1)

    T = Flux.unsqueeze(interp.x, 2) .- Flux.unsqueeze(interp.x, 1)
    d = T .+ interp.mask
    denom = Flux.prod(d ./ scale .+ 1f-5, dims = 1)
    coef = numer ./ denom
    return dropdims(sum(interp.y .* coef, dims = 2), dims = 2)
end

function linear_interp(X::AbstractMatrix{Float32}, Y::AbstractArray{Float32, 3}, x::AbstractVector{Float32})
    x_row = permutedims(x)

    d = X .- x_row
    ΔYΔX = diff(Y, dims = 2) ./ Flux.unsqueeze(diff(d, dims = 1), 1)
    l = X[1:end-1, :]
    r = X[2:end, :]

    ## short circut evaluation used to cover edge case when x is the final X point
    final_step = r .== r[[end], :] .== x_row
    mask = (l .<= x_row .< r) .|| final_step

    x0 = sum(X[1:end-1, :] .* mask, dims = 1)
    y0 = dropdims(sum(Y[:, 1:end-1, :] .* Flux.unsqueeze(mask, 1), dims = 2), dims = 2)
    dydx = dropdims(sum(ΔYΔX .* Flux.unsqueeze(mask, 1), dims = 2), dims = 2)

    return y0 .+ (permutedims(x) .- x0) .* dydx
end

struct LinearInterpolation
    X::AbstractMatrix{Float32}
    Y::AbstractArray{Float32, 3}
end

Flux.@functor LinearInterpolation

function (interp::LinearInterpolation)(x::AbstractVector)
    return linear_interp(interp.X, interp.Y, x)
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
        latent_dim::OneDim
        )

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
        x -> hcat(x[:, [1], :], x[:, [2], :] ./ c0, x[:, [3], :], x[:, [4], :] ./ c0)
        )
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
    actions = 20))

policy = RandomDesignPolicy(action_space(env))
# render!(policy, env, path = "vid.mp4")
ep = generate_episode!(policy, env)

horizon = 5
data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = 4, shuffle = true, partial = false)
s, a, t, y = gpu(Flux.batch.(first(data)))
t_ = gpu(t[1:env.integration_steps:end, :])

latent_dim = OneDim(15.0f0, 700)
nfreq = 50
emb = Waves.SinWaveEmbedder(latent_dim, nfreq)
w = gpu(emb(randn(Float32, nfreq, size(t_)...)))

C = gpu(Chain(LinearInterpolation(t_, w), sigmoid))
F = gpu(Waves.SinusoidalSource(latent_dim, nfreq, 1000.0f0))
θ = [C, F]

dyn = gpu(AcousticDynamics(latent_dim, WATER, 5.0f0, 10000.0f0))
iter = gpu(Integrator(runge_kutta, dyn, env.dt))
wave_encoder = gpu(build_wave_encoder(;latent_dim, nfreq))
wave = wave_encoder(s)




# tx = t[150, :]
# y_hat, back = Flux.pullback(w) do _w
#     linear_interp(t_, _w, tx)
# end
# gs = back(y_hat)[1]




# """
# testing differentiation through simulation
# """
# z, back = Flux.pullback(θ) do _θ
#     return iter(wave, t, _θ)
# end
# @time gs = cpu(back(z))[1]

# gs_C = cpu(gs[1][1][1].Y)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, latent_dim.x, gs_C[:, 1, 1], label = "1")
# lines!(ax, latent_dim.x, gs_C[:, 2, 1], label = "2")
# lines!(ax, latent_dim.x, gs_C[:, 3, 1], label = "3")
# lines!(ax, latent_dim.x, gs_C[:, 4, 1], label = "4")
# lines!(ax, latent_dim.x, gs_C[:, 5, 1], label = "5")
# lines!(ax, latent_dim.x, gs_C[:, 6, 1], label = "6")
# axislegend(ax)
# save("gs_C.png", fig)






# """
# plotting gradients
# """
# # fig = Figure()
# # ax = Axis(fig[1, 1])
# # lines!(ax, gs[1][1][1].y[:, 1, 1], label = "1")
# # lines!(ax, gs[1][1][1].y[:, 2, 1], label = "2")
# # lines!(ax, gs[1][1][1].y[:, 3, 1], label = "3")
# # lines!(ax, gs[1][1][1].y[:, 4, 1], label = "4")
# # lines!(ax, gs[1][1][1].y[:, 5, 1], label = "5")
# # lines!(ax, gs[1][1][1].y[:, 6, 1], label = "6")
# # axislegend(ax)
# # save("gs.png", fig)






# """
# testing single timestep differentiation
# """
# # θ = [C, F]
# # # for i in axes(t, 1)
# #     dwave, back = Flux.pullback(θ) do _θ
# #         return dyn(wave, t[end, :], _θ)
# #     end
    
# #     gs = back(dwave)[1]
# #     # display(i)
# # # end



# """
# Testing with known function interp
# """
# y = sin.(1000.0f0 .* t_)
# y_true = sin.(1000.0f0 .* t)
# # interp = gpu(PolynomialInterpolation(t′, Flux.unsqueeze(y, 1)))
# # y_pred = vcat([interp(t[i, :]) for i in axes(t, 1)]...)
# y_pred = vcat([linear_interp(t_, Flux.unsqueeze(y, 1), t[i, :]) for i in axes(t, 1)]...)

# fig = Figure()
# ax = Axis(fig[1, 1])
# scatter!(ax, cpu(t_[:, 1]), cpu(y[:, 1]))
# lines!(ax, cpu(t[:, 1]), cpu(y_pred[:, 1]), color = :green)
# lines!(ax, cpu(t[:, 1]), cpu(y_true[:, 1]), color = :orange)
# save("interp.png", fig)

# """
# Testing AD vs AS
# """
# # z, back = Flux.pullback(wave) do _wave
# #     z = iter(_wave, t, θ)
# # end

# # adj = z .^ 0.0f0

# # @time gs_AD = cpu(back(adj)[1])
# # @time gs_AS = cpu(adjoint_sensitivity(iter, z, t, θ, adj))

# # fig = Figure()
# # ax1 = Axis(fig[1, 1])
# # ax2 = Axis(fig[1, 1])
# # ax2.yaxisposition = :right
# # ax3 = Axis(fig[1, 2])
# # ax4 = Axis(fig[1, 2])
# # ax4.yaxisposition = :right

# # lines!(ax1, gs_AD[:, 1, 1], color = :blue)
# # lines!(ax2, gs_AS[:, 1, 1], color = :orange)
# # lines!(ax3, gs_AD[:, 2, 1], color = :blue)
# # lines!(ax4, gs_AS[:, 2, 1], color = :orange)
# # save("gs.png", fig)

# """
# rendering simulation
# """
# z = cpu(iter(wave, t, θ))

# fig = Figure()
# ax = Axis(fig[1, 1])
# xlims!(ax, latent_dim.x[1], latent_dim.x[end])
# ylims!(ax, -1.0f0, 1.0f0)

# record(fig, "interp.mp4", axes(t, 1)) do i
#     c = cpu(C(t[i, :]))
#     f = cpu(F(t[i, :]))

#     empty!(ax)
#     lines!(ax, latent_dim.x, c[:, 1], color = :red)
#     lines!(ax, latent_dim.x, f[:, 1], color = :green)
#     lines!(ax, latent_dim.x, z[:, 1, 1, i], color = :blue)
# end
