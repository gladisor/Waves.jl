using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie: heatmap!, save
using Waves

include("wave_net.jl")
# include("wave_control_net.jl")
include("wave_encoder.jl")

function Base.vec(config::Scatterers)
    return vcat(vec(config.pos), config.r, config.c)
end

function Waves.DesignTrajectory(design::DesignInterpolator, n::Int)

    t = collect(range(design.ti, design.tf, n + 1))
    traj = typeof(design.initial)[]

    for i ∈ axes(t, 1)
        push!(traj, design(t[i]))
    end

    return DesignTrajectory(traj)
end

Flux.@functor Scatterers

grid_size = 5.0f0
elements = 256
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

config = Scatterers([0.0f0 0.0f0], [0.5f0], [0.5f0])

env = gpu(WaveEnv(
    initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    space = design_space(config, 1.0f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))
traj = episode_trajectory(env)
agent = Agent(policy, traj)
@time run(agent, env, StopWhenDone())

function Waves.render!(traj::Trajectory; path::String)
    states = traj.traces.state[2:end]
    actions = traj.traces.action[1:end-1]

    design = DesignTrajectory[]

    for (s, a) ∈ zip(states, actions)
        interp = DesignInterpolator(s.design, a, s.sol.total.t[1], s.sol.total.t[end])
        dt = DesignTrajectory(interp, length(s.sol.total)-1)
        push!(design, dt)
    end

    sol = WaveSol([s.sol.total for s ∈ states]...)
    design = DesignTrajectory(design...)

    render!(sol, design, path =  path)
end

render!(traj, path = "vid.mp4")

# encoder = WaveEncoder(fields, 2, 2, tanh)
# z = encoder(s.sol.total)
# latents = integreate(cell, z, z_dynamics, 10)

# # # states = traj.traces.state[2:end]
# # # actions = traj.traces.action[1:end-1]

# # # sol = WaveSol([s.sol.total for s ∈ states]...)
# # # design = DesignTrajectory([s.design for s ∈ states])

# # # wave_states = traj.traces.state[2:end]
# # # design_images = [Waves.speed(design, env.total_dynamics.g, env.total_dynamics.ambient_speed) for design in design_states.states]
# # # a = traj.traces.action[1:end-1]

# # # steps = 100
# # # cell = WaveCell(split_wave_pml, runge_kutta)
# # # ic = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
# # # wave = ic(build_wave(dim, fields = 6))
# # # dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)
# # # @time sol = solve(cell, wave, dynamics, steps) |> gpu
# # # render!(sol, path = "vid.mp4")

# # # z_elements = (elements ÷ (2^3)) ^ 2

# # # layers = Chain(
# # #     Dense(z_elements, z_elements * 2, relu),
# # #     Dense(z_elements * 2, z_elements * 2, relu),
# # #     z -> sum(z, dims = 2),
# # #     Dense(z_elements * 2, z_elements, sigmoid))

# # # cell = WaveRNNCell(nonlinear_latent_wave, runge_kutta, layers)
# # # net = WaveNet(
# # #     grid_size = grid_size, 
# # #     elements = elements,
# # #     cell = WaveCell(split_wave_pml, runge_kutta),
# # #     fields = fields, 
# # #     h_fields = 32, 
# # #     z_fields = 2,
# # #     activation = relu;
# # #     dynamics_kwargs...) |> gpu

# # # ps = Flux.params(net)
# # # opt = Adam(0.0005)

# # # u = cat(sol.u[2:end]..., dims = 4) |> gpu

# # # for epoch ∈ 1:100
# # #     Waves.reset!(net.z_dynamics)

# # #     gs = Flux.gradient(ps) do 

# # #         y = net(sol)
# # #         loss = sqrt(Flux.Losses.mse(u, y))

# # #         Flux.ignore() do 
# # #             println(loss)
# # #         end

# # #         return loss
# # #     end

# # #     Flux.Optimise.update!(opt, ps, gs)
# # # end

# # # p = WavePlot(dim)
# # # heatmap!(p.ax, dim.x, dim.y, cpu(net(sol)[:, :, 1, end]), colormap = :ice)
# # # save("u.png", p.fig)