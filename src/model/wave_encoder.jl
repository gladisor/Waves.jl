
"""
Builds CNN feature extractor for processing WaveEnvState(s).
"""
function build_cnn_base(env::WaveEnv, in_channels::Int, activation::Function, out_channels::Int)
    return Chain(
        TotalWaveInput(),
        LocalizationLayer(env.dim, env.resolution),
        ResidualBlock((3, 3), 2 + in_channels, 32, activation),
        ResidualBlock((3, 3), 32, 64, activation),
        ResidualBlock((3, 3), 64, out_channels, activation),
        GlobalMaxPool(),
        Flux.flatten)
end

struct WaveEncoder
    base::Chain
    head::Chain
end

Flux.@functor WaveEncoder

function build_wave_encoder_head(env::WaveEnv, h_size::Int, activation::Function, nfreq::Int, latent_dim::OneDim, pml_func::Function = x -> x ^ 2)
    return Chain(
        Parallel(
            vcat,
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq)),
            Chain(Dense(h_size, h_size, activation), Dense(h_size, h_size, activation), Dense(h_size, nfreq))
            ),
        b -> reshape(b, nfreq, 6, :),
        SinWaveEmbedder(latent_dim, nfreq),
        x -> hcat(
            x[:, [1], :],       # u_tot
            x[:, [2], :] ./ env.iter.dynamics.c0, # v_tot
            x[:, [3], :],       # u_inc
            x[:, [4], :] ./ env.iter.dynamics.c0, # v_inc
            x[:, [5], :],       # f
            pml_func.(x[:, [6], :])
            )
        )
end

function WaveEncoder(env::WaveEnv, h_size::Int, activation::Function, nfreq::Int, latent_dim::OneDim)
    base = build_cnn_base(env, 3, activation, h_size)
    head = build_wave_encoder_head(env, h_size, activation, nfreq, latent_dim)
    return WaveEncoder(base, head)
end

function (enc::WaveEncoder)(s::Vector{WaveEnvState})
    return s |> enc.base |> enc.head
end