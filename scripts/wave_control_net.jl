
struct WaveControlNet
    z_elements::Int
    z_fields::Int

    cell::AbstractWaveCell
    z_dynamics::WaveDynamics

    down1::DownBlock
    down2::DownBlock
    down3::DownBlock

    control_state_embed::Chain
    control_action_embed::Chain

    up1::UpBlock
    up2::UpBlock
    up3::UpBlock
    out::Conv
end

Flux.@functor WaveControlNet (down1, down2, down3, control_state_embed, control_action_embed, up1, up2, up3)

function WaveControlNet(;grid_size::Float32, elements::Int, cell::AbstractWaveCell, fields::Int, z_fields::Int, h_fields::Int, activation::Function, dynamics_kwargs...)
    z_elements = (elements ÷ (2^3)) ^ 2
    z_dim = OneDim(grid_size, z_elements)
    z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...)

    down1 = DownBlock(3, fields, h_fields, activation)
    down2 = DownBlock(3, h_fields, h_fields, activation)
    down3 = DownBlock(3, h_fields, z_fields, activation)

    control_state_embed = Chain(
        vec,
        Dense(4, 128, activation),
        Dense(128, 128, activation),
        Dense(128, z_fields, activation))

    control_action_embed = Chain(
        vec,
        Dense(4, 128, activation),
        Dense(128, 128, activation),
        Dense(128, z_fields, activation))

    up1 = UpBlock(3, z_fields + 2, h_fields, activation)
    up2 = UpBlock(3, h_fields, h_fields, activation)
    up3 = UpBlock(3, h_fields, h_fields, activation)
    out = Conv((3, 3), h_fields => fields, tanh, pad = SamePad())

    return WaveControlNet(z_elements, z_fields, cell, z_dynamics, down1, down2, down3, control_state_embed, control_action_embed, up1, up2, up3, out)
end


