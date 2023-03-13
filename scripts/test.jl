using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using CairoMakie: heatmap!, save
using Waves

include("wave_net.jl")

grid_size = 5.0f0
elements = 128
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

cyl = Cylinder(0.0f0, 0.0f0, 0.5f0, 0.5f0)

env = gpu(WaveEnv(
    initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = cyl,
    space = design_space(cyl, 1.0f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

s = state(env)

policy = RandomDesignPolicy(action_space(env))
traj = episode_trajectory(env)
agent = Agent(policy, traj)

# design_states = DesignStates()
# data = SaveData()

# hook = ComposedHook(
#     design_states,
#     data)

@time run(agent, env, StopWhenDone())

states = traj.traces.state[2:end]
actions = traj.traces.action[1:end-1]

# wave_states = traj.traces.state[2:end]
# design_images = [Waves.speed(design, env.total_dynamics.g, env.total_dynamics.ambient_speed) for design in design_states.states]
# a = traj.traces.action[1:end-1]

# steps = 100
# cell = WaveCell(split_wave_pml, runge_kutta)
# ic = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)
# wave = ic(build_wave(dim, fields = 6))
# dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)
# @time sol = solve(cell, wave, dynamics, steps) |> gpu
# render!(sol, path = "vid.mp4")

# z_elements = (elements ÷ (2^3)) ^ 2

# layers = Chain(
#     Dense(z_elements, z_elements * 2, relu),
#     Dense(z_elements * 2, z_elements * 2, relu),
#     z -> sum(z, dims = 2),
#     Dense(z_elements * 2, z_elements, sigmoid))

# cell = WaveRNNCell(nonlinear_latent_wave, runge_kutta, layers)
# net = WaveNet(
#     grid_size = grid_size, 
#     elements = elements,
#     cell = WaveCell(split_wave_pml, runge_kutta),
#     fields = fields, 
#     h_fields = 32, 
#     z_fields = 2,
#     activation = relu;
#     dynamics_kwargs...) |> gpu

# ps = Flux.params(net)
# opt = Adam(0.0005)

# u = cat(sol.u[2:end]..., dims = 4) |> gpu

# for epoch ∈ 1:100
#     Waves.reset!(net.z_dynamics)

#     gs = Flux.gradient(ps) do 

#         y = net(sol)
#         loss = sqrt(Flux.Losses.mse(u, y))

#         Flux.ignore() do 
#             println(loss)
#         end

#         return loss
#     end

#     Flux.Optimise.update!(opt, ps, gs)
# end

# p = WavePlot(dim)
# heatmap!(p.ax, dim.x, dim.y, cpu(net(sol)[:, :, 1, end]), colormap = :ice)
# save("u.png", p.fig)