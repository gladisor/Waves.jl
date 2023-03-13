struct DownBlock
    conv1::Conv
    conv2::Conv
    conv3::Conv
    pool::MaxPool
end

Flux.@functor DownBlock

function DownBlock(k, in_channels, out_channels, activation)

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

function UpBlock(k, in_channels, out_channels, activation)

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

struct WaveNet
    z_elements::Int
    z_fields::Int

    cell::AbstractWaveCell
    z_dynamics::WaveDynamics

    down1::DownBlock
    down2::DownBlock
    down3::DownBlock

    up1::UpBlock
    up2::UpBlock
    up3::UpBlock
    out::Conv
end

Flux.@functor WaveNet (down1, down2, down3, up1, up2, up3)

function WaveNet(;grid_size::Float32, elements::Int, cell::AbstractWaveCell, fields::Int, z_fields::Int, h_fields::Int, activation::Function, dynamics_kwargs...)

    z_elements = (elements ÷ (2^3)) ^ 2
    z_dim = OneDim(grid_size, z_elements)
    z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...)

    down1 = DownBlock(3, fields, h_fields, activation)
    down2 = DownBlock(3, h_fields, h_fields, activation)
    down3 = DownBlock(3, h_fields, z_fields, activation)

    up1 = UpBlock(3, z_fields, h_fields, activation)
    up2 = UpBlock(3, h_fields, h_fields, activation)
    up3 = UpBlock(3, h_fields, h_fields, activation)
    out = Conv((3, 3), h_fields => fields, tanh, pad = SamePad())

    return WaveNet(z_elements, z_fields, cell, z_dynamics, down1, down2, down3, up1, up2, up3, out)
end

function (net::WaveNet)(x::AbstractArray{Float32, 3}, steps::Int)
    x1 = net.down1(Flux.batch([x]))
    x2 = net.down2(x1)
    x3 = net.down3(x2)

    z = reshape(x3, net.z_elements, net.z_fields)
    z_wave = cat(integrate(net.cell, z, net.z_dynamics, steps)..., dims = 3)
    n = Int(sqrt(net.z_elements))
    
    y = reshape(z_wave, n, n, net.z_fields, steps)
    y1 = net.up1(y)
    y2 = net.up2(y1)
    y3 = net.up3(y2)

    return net.out(y3)
end

function (net::WaveNet)(sol::WaveSol)
    steps = length(sol) - 1
    return net(first(sol.u), steps)
end

function Flux.gpu(net::WaveNet)
    return WaveNet(
        net.z_elements, 
        net.z_fields,
        gpu(net.cell),
        gpu(net.z_dynamics),
        gpu(net.down1),
        gpu(net.down2),
        gpu(net.down3),
        gpu(net.up1),
        gpu(net.up2),
        gpu(net.up3),
        gpu(net.out)
        )
end

function Flux.cpu(net::WaveNet)
    return WaveNet(
        net.z_elements,
        net.z_fields,
        cpu(net.cell),
        cpu(net.z_dynamics),
        cpu(net.down1),
        cpu(net.down2),
        cpu(net.down3),
        cpu(net.up1),
        cpu(net.up2),
        cpu(net.up3),
        cpu(net.out)
        )
end