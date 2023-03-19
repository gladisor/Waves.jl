export WaveDecoder

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