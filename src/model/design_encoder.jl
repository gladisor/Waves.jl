export DesignEncoder

"""
Normalizes the design parameter vector between -1 and 1
"""
function normalize(design::AbstractDesign, ds::DesignSpace)
    scale = 2.0f0
    return scale * (vec(design) .- vec(ds.low)) ./ (vec(ds.high) .- vec(ds.low) .+ 1f-3) .- (scale / 2.0f0)
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
    x = normalize.(design_sequences, [de.design_space])
    x_batch = cat([hcat(x[i, :]...) for i in axes(x, 1)]..., dims = 3)
    y = de.layers(x_batch)
    return LinearInterpolation(t_, y)
end

function DesignEncoder(env::WaveEnv, h_size::Int, activation::Function, nfreq::Int, latent_dim::OneDim)
    mlp = Chain(
        Dense(length(vec(env.design)), h_size, activation), 
        Dense(h_size, h_size, activation), 
        Dense(h_size, h_size, activation),
        Dense(h_size, h_size, activation), 
        Dense(h_size, nfreq),
        SinWaveEmbedder(latent_dim, nfreq),
        c -> 2.0f0 * sigmoid.(c))
    return DesignEncoder(env.design_space, mlp, env.integration_steps)
end