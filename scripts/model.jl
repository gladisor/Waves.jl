using Flux

mutable struct WaveNet
    domain
    encoder
    Ψ
    restructure
end

Flux.@functor WaveNet

Flux.trainable(wn::WaveNet) = (wn.encoder, wn.Ψ)

function WaveNet(sim::WaveSim{OneDim};
        in_size, enc_h_size, enc_n_hidden, z_size,
        Φ_h_size, Φ_n_hidden, Ψ_h_size, Ψ_n_hidden, σ)

    x = dims(sim)[1]

    encoder = Chain(
        Dense(in_size, enc_h_size, σ),
        [Dense(enc_h_size, enc_h_size, σ) for _ ∈ enc_n_hidden]...,
        Dense(enc_h_size, z_size))

    Φ = Chain(
        Dense(1, Φ_h_size, σ),
        [Dense(Φ_h_size, Φ_h_size, σ) for _ ∈ Φ_n_hidden]...,
        Dense(Φ_h_size, 1))

    Φ_θ, restructure = Flux.destructure(Φ)

    Ψ = Chain(
        Dense(z_size, Ψ_h_size, σ),
        [Dense(Ψ_h_size, Ψ_h_size, σ) for _ ∈ Ψ_n_hidden]...,
        Dense(Ψ_h_size, length(Φ_θ), bias = false))

    return WaveNet(x, encoder, Ψ, restructure)
end

function (wn::WaveNet)(x)
    z = wn.encoder(x) ## embedding signal in z space
    θ = wn.Ψ(z) ## constructing weights θ of Φ network using hypernet
    Φ = wn.restructure.(eachcol(θ)) ## restructuring weights into Φ
    I = vcat(map(Φ -> Φ(wn.domain'), Φ)...)' ## evaluating Φ on the domain
    return I
end

# function (wn::WaveNet)(ws::WaveSolution1D)

#     u0, _ = time_data(ws)
#     u′ = wn(u0 |> gpu) |> cpu

#     return WaveSolution1D(ws.x, ws.t, hcat(u0[1:end-1, 1], u′))
# end