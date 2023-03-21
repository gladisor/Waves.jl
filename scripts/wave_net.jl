struct WaveNet
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    z_cell::AbstractWaveCell
    z_dynamics::WaveDynamics
    wave_decoder::WaveDecoder
end

Flux.@functor WaveNet (wave_encoder, design_encoder, wave_decoder, cell)

function (model::WaveNet)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign, steps::Int)
    z = hcat(model.wave_encoder(wave), model.design_encoder(design, action))
    b = model.design_encoder(design, action)
    latents = integrate(model.z_cell, cat(z, b, dims = 2), model.z_dynamics, steps)
    z_concat = cat(latents..., dims = 3)
    n = Int(sqrt(size(z_concat, 1)))
    z_feature_maps = reshape(z_concat, n, n, size(z_concat, 2), :)
    return model.wave_decoder(z_feature_maps)
end

function (model::WaveNet)(s::WaveEnvState, action::AbstractDesign)
    return model(s.sol.total.u[1], s.design, action, length(s.sol.total) - 1)
end


function Flux.gpu(model::WaveNet)
    return WaveNet(
        gpu(model.wave_encoder),
        gpu(model.design_encoder),
        model.z_cell,
        gpu(model.z_dynamics),
        gpu(model.wave_decoder),
    )
end

function Flux.cpu(model::WaveNet)
    return WaveNet(
        cpu(model.wave_encoder),
        cpu(model.design_encoder),
        model.z_cell,
        cpu(model.z_dynamics),
        cpu(model.wave_decoder),
    )
end

function loss(model::WaveNet, s::WaveEnvState, action::AbstractDesign)
    u_true = get_target_u(s.sol.incident)
    u_pred = model(s, action)
    return sqrt(mse(u_true, u_pred))
end