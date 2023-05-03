export DownBlock, UpBlock
export DesignEncoder
export WaveEncoder, WaveDecoder
export WaveControlModel, encode

struct DownBlock
    conv1::Conv
    conv2::Conv
    conv3::Conv
    pool::MaxPool
end

Flux.@functor DownBlock

function DownBlock(k::Int, in_channels::Int, out_channels::Int, activation::Function)

    conv1 = Conv((k, k), in_channels => out_channels, activation, pad = SamePad())
    conv2 = Conv((k, k), out_channels => out_channels, activation, pad = SamePad())
    conv3 = Conv((k, k), out_channels => out_channels, activation, pad = SamePad())
    pool = MaxPool((2, 2))

    return DownBlock(conv1, conv2, conv3, pool)
end

function (block::DownBlock)(x)
    x = x |> block.conv1 |> block.conv2 |> block.conv3
    return block.pool(x)
end

struct UpBlock
    conv1::Conv
    conv2::Conv
    conv3::Conv
    upsample::Upsample
end

Flux.@functor UpBlock

function UpBlock(k::Int, in_channels::Int, out_channels::Int, activation::Function)

    conv1 = Conv((k, k), in_channels => out_channels, activation, pad = SamePad())
    conv2 = Conv((k, k), out_channels => out_channels, activation, pad = SamePad())
    conv3 = Conv((k, k), out_channels => out_channels, activation, pad = SamePad())
    upsample = Upsample((2, 2))

    return UpBlock(conv1, conv2, conv3, upsample)
end

function (block::UpBlock)(x)
    x = x |> block.conv1 |> block.conv2 |> block.conv3
    return block.upsample(x)
end

struct DesignEncoder
    dense1::Dense
    dense2::Dense
    dense3::Dense
    dense4::Dense
end

Flux.@functor DesignEncoder

function DesignEncoder(in_size::Int, h_size::Int, out_size::Int, activation::Function)
    dense1 = Dense(in_size, h_size, activation)
    dense2 = Dense(h_size, h_size, activation)
    dense3 = Dense(h_size, h_size, activation)
    dense4 = Dense(h_size, out_size, sigmoid)
    return DesignEncoder(dense1, dense2, dense3, dense4)
end

function (encoder::DesignEncoder)(design::AbstractDesign, action::AbstractDesign)
    x = vcat(vec(design), vec(action))
    return x |> encoder.dense1 |> encoder.dense2 |> encoder.dense3 |> encoder.dense4
end

struct WaveEncoder
    down1::DownBlock
    down2::DownBlock
    down3::DownBlock
end

Flux.@functor WaveEncoder

function WaveEncoder(fields::Int, h_fields::Int, z_fields::Int, activation::Function)

    down1 = DownBlock(3, fields,   h_fields, activation)
    down2 = DownBlock(3, h_fields, h_fields, activation)
    down3 = DownBlock(3, h_fields, z_fields, activation)

    return WaveEncoder(down1, down2, down3)
end

function (encoder::WaveEncoder)(wave::AbstractArray{Float32, 3})
    z = encoder.down1(wave[:, :, :, :])
    z = encoder.down2(z)
    z = encoder.down3(z)
    return Flux.flatten(z[:, :, :, 1])
end

struct WaveDecoder
    up1::UpBlock
    up2::UpBlock
    up3::UpBlock
    out::Conv
end

Flux.@functor WaveDecoder

function WaveDecoder(fields::Int, h_fields::Int, z_fields::Int, activation::Function)

    up1 = UpBlock(3, z_fields,   h_fields, activation)
    up2 = UpBlock(3, h_fields,   h_fields, activation)
    up3 = UpBlock(3, h_fields,   h_fields,   activation)
    out = Conv((3, 3), h_fields => fields, tanh, pad = SamePad())

    return WaveDecoder(up1, up2, up3, out)
end

function (decoder::WaveDecoder)(z::AbstractArray{Float32, 4})
    z = decoder.up1(z)
    z = decoder.up2(z)
    z = decoder.up3(z)
    return decoder.out(z)
end

struct WaveControlModel <: AbstractWaveControlModel
    wave_encoder::Chain
    design_encoder::Chain
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function (model::WaveControlModel)(h::Tuple{AbstractMatrix{Float32}, AbstractDesign}, action::AbstractDesign)
    z_wave, design = h
    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    z = model.iter(hcat(z_wave, z_design))
    sigma = model.mlp(z)
    return (z[:, [1, 2], end], design + action), sigma
end

function (model::WaveControlModel)(s::WaveEnvState, actions::Vector{<:AbstractDesign})
    z_wave = model.wave_encoder(s.wave_total)
    recur = Recur(model, (z_wave, s.design))
    return hcat([recur(action) for action in actions]...)
end

function (model::WaveControlModel)(s::WaveEnvState, action::AbstractDesign)
    return vec(model(s, [action]))
end

function encode(model::WaveControlModel, s::WaveEnvState, action::AbstractDesign)
    z_wave = model.wave_encoder(s.wave_total)
    z_design = model.design_encoder(vcat(vec(s.design), vec(action)))
    return hcat(z_wave, z_design)
end