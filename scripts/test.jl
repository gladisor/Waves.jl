using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using IntervalSets
using CairoMakie
using Statistics: mean
using Flux.Data: DataLoader

using Waves

include("wave_encoder.jl")
include("design_encoder.jl")
include("wave_decoder.jl")

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv)
    traj = episode_trajectory(env)
    agent = Agent(policy, traj)
    run(agent, env, StopWhenDone())

    states = traj.traces.state[2:end]
    actions = traj.traces.action[1:end-1]

    return (states, actions)
end

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv, episodes::Int)
    states = []
    actions = []

    for _ ∈ 1:episodes
        s, a = generate_episode_data(policy, env)
        push!(states, s)
        push!(actions, a)
    end

    return (vcat(states...), vcat(actions...))
end

function random_radii_scatterer_formation()
    config = scatterer_formation(width = 1, hight = 2, spacing = 1.0f0, r = 0.5f0, c = 0.20f0, center = [2.0f0, 0.0f0])
    r = rand(Float32, size(config.r))
    r = r * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    return Scatterers(config.pos, r, config.c)
end

config = random_radii_scatterer_formation()
grid_size = 5.0f0
elements = 256
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    initial_condition = PlaneWave(dim, -4.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    random_design = random_radii_scatterer_formation,
    space = radii_design_space(config, 0.2f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))

@time render!(policy, env, path = "vid.mp4")

# @time train_data = generate_episode_data(policy, env, 50)
# @time test_data = generate_episode_data(policy, env, 5)
# @time val_data = generate_episode_data(policy, env, 1);

# train_data_loader = DataLoader(train_data, shuffle = true)
# test_data_loader = DataLoader(test_data, shuffle = true)

# h_fields = 32
# z_fields = 2
# activation = relu
# z_size = Int.(size(dim) ./ (2 ^ 3))
# design_size = 2 * length(vec(config))

# wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation) |> gpu
# design_encoder = DesignEncoder(design_size, 128, prod(z_size), activation) |> gpu
# cell = WaveCell(nonlinear_latent_wave, runge_kutta)
# z_dim = OneDim(grid_size, prod(z_size))
# z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...) |> gpu
# wave_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation) |> gpu

# opt = Adam(0.0001)
# ps = Flux.params(wave_encoder, wave_decoder, design_encoder)
# train_loss = Float32[]
# test_loss = Float32[]

# for epoch in 1:100
#     for (s, a) in train_data_loader
#         s = gpu(first(s))
#         a = gpu(first(a))

#         u_true = cat(s.sol.total.u[2:end]..., dims = 4)

#         gs = Flux.gradient(ps) do 
#             z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
#             latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
#             z_sequence = reshape(latents, z_size..., z_fields + 1, env.design_steps)
#             u_pred = wave_decoder(z_sequence)

#             loss = sqrt(Flux.Losses.mse(u_true, u_pred))

#             Flux.ignore() do
#                 push!(train_loss, loss)
#                 println("Train Loss: $loss")
#             end

#             return loss
#         end

#         Flux.Optimise.update!(opt, ps, gs)
#     end

#     batch_test_loss = []

#     for (s, a) in test_data_loader
#         s = gpu(first(s))
#         a = gpu(first(a))

#         u_true = cat(s.sol.total.u[2:end]..., dims = 4)
        
#         z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
#         latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
#         z_sequence = reshape(latents, z_size..., z_fields + 1, env.design_steps)
#         u_pred = wave_decoder(z_sequence)

#         loss = sqrt(Flux.Losses.mse(u_true, u_pred))
#         push!(batch_test_loss, loss)
#     end

#     for (i, (s, a)) in enumerate(zip(val_data...))

#         s = gpu(s)
#         a = gpu(a)

#         u_true = cat(s.sol.total.u[2:end]..., dims = 4)
#         z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
#         latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
#         z_sequence = reshape(latents, z_size..., z_fields + 1, env.design_steps)
#         u_pred = wave_decoder(z_sequence)

#         Waves.plot_comparison!(dim, cpu(u_true), cpu(u_pred), path = "comparison_$(i).png")
#     end

#     push!(test_loss, mean(batch_test_loss))

#     fig = Figure()
#     ax1 = Axis(fig[1, 1])
#     ax2 = Axis(fig[1, 2])
#     lines!(ax1, train_loss, color = :blue)
#     scatter!(ax2, test_loss, color = :red)
#     save("loss.png", fig)
# end