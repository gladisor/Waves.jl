struct WaveControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function encode(model::WaveControlModel, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    u = model.wave_encoder(wave)
    v = u * 0.0f0
    return hcat(u, v, model.design_encoder(design, action))
end

function (model::WaveControlModel)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    zi = encode(model, wave, design, action)
    z = model.iter(zi)
    return model.mlp(z)
end

function build_control_sequence(action::AbstractDesign, steps::Int)
    return [zero(action) for i in 1:steps]
end

function build_mpc_cost(model::WaveControlModel, s::ScatteredWaveEnvState, control_sequence::Vector{ <: AbstractDesign})
    cost = 0.0f0

    d1 = s.design
    c1 = model.design_encoder(d1, control_sequence[1])
    z1 = hcat(model.wave_encoder(s.wave_total), c1)
    z = model.iter(z1)
    cost = cost + sum(model.mlp(z))

    d2 = d1 + control_sequence[1]
    c2 = model.design_encoder(d2, control_sequence[2])
    z2 = hcat(z[:, 1:2, end], c2)
    z = model.iter(z2)
    cost = cost + sum(model.mlp(z))

    d3 = d2 + control_sequence[2]
    c3 = model.design_encoder(d3, control_sequence[3])
    z3 = hcat(z[:, 1:2, end], c3)
    z = model.iter(z3)
    cost = cost + sum(model.mlp(z))

    return cost
end