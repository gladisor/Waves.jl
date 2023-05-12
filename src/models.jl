export DownBlock, UpBlock
export DesignEncoder
export WaveEncoder, WaveDecoder
export WaveControlModel, propagate, encode_design, encode

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

function encode_design(model::WaveControlModel, design::AbstractDesign, a::AbstractDesign, scale::Float32 = 0.25f0)
    return model.design_encoder(vcat(vec(design), vec(a) / scale))
end

function propagate(model::WaveControlModel, z_wave::AbstractMatrix{Float32}, design::AbstractDesign, action::AbstractDesign)
    z_design = encode_design(model, design, action)
    z = model.iter(hcat(z_wave, z_design))
    return (z, z[:, [1, 2, 3], end], design + action)
end

function (model::WaveControlModel)(h::Tuple{AbstractMatrix{Float32}, AbstractDesign}, action::AbstractDesign)
    z_wave, design = h
    z, z_wave, design = propagate(model, z_wave, design, action)
    sigma = model.mlp(z)
    return (z_wave, design), sigma
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
    z_design = encode_design(model, s.design, action)
    return hcat(z_wave, z_design)
end

function FileIO.save(model::WaveControlModel, path::String)
    BSON.bson(joinpath(path, "wave_encoder.bson"), wave_encoder = cpu(model.wave_encoder))
    BSON.bson(joinpath(path, "design_encoder.bson"), design_encoder = cpu(model.design_encoder))
    BSON.bson(joinpath(path, "iter.bson"), iter = cpu(model.iter))
    BSON.bson(joinpath(path, "mlp.bson"), mlp = cpu(model.mlp))
end

function WaveControlModel(;path::String)
    wave_encoder = BSON.load(joinpath(path, "wave_encoder.bson"))[:wave_encoder]
    design_encoder = BSON.load(joinpath(path, "design_encoder.bson"))[:design_encoder]
    iter = BSON.load(joinpath(path, "iter.bson"))[:iter]
    mlp = BSON.load(joinpath(path, "mlp.bson"))[:mlp]
    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end