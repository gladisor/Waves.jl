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

# using Flux.ChainRulesCore: Tangent, ZeroTangent
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
