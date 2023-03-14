using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using IntervalSets
using CairoMakie
using Statistics: mean

using Waves
using Flux.Data: DataLoader

include("wave_encoder.jl")
include("wave_decoder.jl")

function scatterer_formation(;width::Int, hight::Int, spacing::Float32, r::Float32, c::Float32, center::Vector{Float32})
    pos = []

    for i ∈ 1:width
        for j ∈ 1:hight
            point = [(i - 1) * (2 * r + spacing), (j - 1) * (2 * r + spacing)]
            push!(pos, point)
        end
    end

    pos = hcat(pos...)'
    pos = (pos .- mean(pos, dims = 1)) .+ center'

    r = ones(Float32, size(pos, 1)) * r
    c = ones(Float32, size(pos, 1)) * c

    return Scatterers(pos, r, c)
end

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv)
    traj = episode_trajectory(env)
    agent = Agent(policy, traj)
    run(agent, env, StopWhenDone())

    states = traj.traces.state[2:end]
    actions = traj.traces.action[1:end-1]

    return [zip(states, actions)...]
end

config = scatterer_formation(
    width = 3, 
    hight = 5, 
    spacing = 0.2f0, 
    r = 0.5f0, 
    c = 0.50f0,
    center = [2.0f0, 0.0f0])

grid_size = 5.0f0
elements = 512
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    initial_condition = PlaneWave(dim, -4.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    space = radii_design_space(config, 0.2f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))
traj = episode_trajectory(env)
agent = Agent(policy, traj)
@time run(agent, env, StopWhenDone())
@time render!(traj, path = "vid.mp4")

# data = generate_episode_data(policy, env)
# states = traj.traces.state[2:end]
# actions = traj.traces.action[1:end-1]

# z_size = Int.(size(dim) ./ (2 ^ 3))
# h_fields = 64 ## wave_encoder
# h_size = 128 ## design_encoder
# z_fields = 2
# activation = relu

# design_size = 2 * length(vec(config))
# cell = WaveCell(nonlinear_latent_wave, runge_kutta)
# z_dim = OneDim(grid_size, prod(z_size))
# z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...) |> gpu

# wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation) |> gpu
# wave_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation) |> gpu
# design_encoder = DesignEncoder(design_size, h_size, prod(z_size), activation) |> gpu

# opt = Adam(0.0001)
# ps = Flux.params(wave_encoder, wave_decoder, design_encoder)

# for _ in 1:1000

#     for (s, a) in zip(states, actions)

#         s, a = gpu(s), gpu(a)
#         u = cat(s.sol.total.u[2:end]..., dims = 4)

#         Waves.reset!(z_dynamics)

#         gs = Flux.gradient(ps) do

#             z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
#             latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
#             u_pred = wave_decoder(reshape(latents, z_size..., z_fields + 1, env.design_steps))
#             loss = sqrt(Flux.Losses.mse(u, u_pred))

#             Flux.ignore() do
#                 println(loss)
#             end

#             return loss
#         end

#         Flux.Optimise.update!(opt, ps, gs)
#     end
# end

# idx = 1
# s = gpu(states[idx])
# action = actions[idx]
# z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, action))
# Waves.reset!(z_dynamics)
# z_sol = solve(cell, z, z_dynamics, env.design_steps) |> cpu
# # render!(z_sol, path = "z_sol.mp4")

# latents = cat(z_sol.u[2:end]..., dims = 3) |> gpu
# u_pred = wave_decoder(reshape(latents, z_size..., z_fields + 1, env.design_steps))
# u_true = cat(s.sol.total.u[2:end]..., dims = 4)
# using CairoMakie
# Waves.plot_comparison!(dim, cpu(u_true), cpu(u_pred), path = "comparison_$idx.png")

