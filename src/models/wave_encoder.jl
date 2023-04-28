export WaveEncoder

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

# function (encoder::WaveEncoder)(sol::WaveSol)
#     return encoder(sol.u[1])
# end