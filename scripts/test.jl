using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie: heatmap!, save

using Waves

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
    out = Conv((3, 3), h_fields => fields, pad = SamePad())

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

grid_size = 5.0f0
elements = 64
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
cyl = Cylinder(0.0f0, 0.0f0, 0.5f0, 0.5f0)
tmax = 10.0f0

# env = WaveEnv(
#     initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
#     wave = build_wave(dim, fields = fields),
#     cell = WaveCell(split_wave_pml, runge_kutta),
#     design = cyl,
#     space = design_space(cyl, 1.0f0),
#     design_steps = 100,
#     tmax = tmax;
#     dim = dim,
#     dynamics_kwargs...) |> gpu

# policy = RandomDesignPolicy(action_space(env))
# traj = episode_trajectory(env)
# agent = Agent(policy, traj)

# design_states = DesignStates()
# data = SaveData()

# hook = ComposedHook(
#     design_states,
#     data)

# @time run(agent, env, StopWhenDone(), hook)

# wave_states = traj.traces.state[2:end]
# design_images = [Waves.speed(design, env.total_dynamics.g, env.total_dynamics.ambient_speed) for design in design_states.states]
# a = traj.traces.action[1:end-1]

cell = WaveCell(split_wave_pml, runge_kutta)
ic = Pulse(dim, -4.0f0, 0.0f0, 10.0f0)
wave = ic(build_wave(dim, fields = 6))
dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)
@time sol = solve(cell, wave, dynamics, 600) |> gpu

z_elements = (elements ÷ (2^3)) ^ 2

layers = Chain(
    Dense(z_elements, z_elements, relu),
    z -> sum(z, dims = 2),
    sigmoid)

net = WaveNet(
    grid_size = grid_size, elements = elements,
    cell = WaveCell(nonlinear_latent_wave, runge_kutta),
    # cell = WaveRNNCell(nonlinear_latent_wave, runge_kutta, layers),
    fields = fields, h_fields = 1, z_fields = 3,
    activation = tanh;
    dynamics_kwargs...) |> gpu

ps = Flux.params(net)
opt = Adam(0.0005)
u = cat(sol.u[2:end]..., dims = 4) |> gpu

for epoch ∈ 1:1000
    Waves.reset!(net.z_dynamics)

    gs = Flux.gradient(ps) do 

        y = net(sol)
        loss = sqrt(Flux.Losses.mse(u, y))

        Flux.ignore() do 
            println(loss)
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)

    p = WavePlot(dim)
    heatmap!(p.ax, dim.x, dim.y, net(sol)[:, :, 1, end])
    save("u.png", p.fig)
end