include("design_encoder.jl")

struct WaveNet
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    wave_decoder::WaveDecoder
    cell::AbstractWaveCell
    z_dynamics::WaveDynamics
end

Flux.@functor WaveNet (wave_encoder, design_encoder, wave_decoder, cell)

function (net::WaveNet)(s::WaveEnvState, a::AbstractDesign)
    z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
    return z
end