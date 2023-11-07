using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
Flux.CUDA.allowscalar(false)
using ReinforcementLearning

println("Loaded Packages")
Flux.device!(1)
display(Flux.device())

include("../src/model/layers.jl")
include("../src/model/wave_encoder.jl")
include("../src/model/design_encoder.jl")
include("../src/model/model.jl")
include("dataset.jl")
include("../src/model/node.jl")
include("../src/model/mpc.jl")

function compute_energy_cost(model::AcousticEnergyModel, s, a, t)
    y_hat = model(s, a, t)
    return vec(sum(y_hat[:, 3, :], dims = 1))
end

function compute_energy_cost(model::NODEEnergyModel, s, a, t)
    y_hat = model(s, a, t)
    return vec(sum(y_hat, dims = 1))
end

function Waves.build_tspan(mpc::RandomShooting, env::WaveEnv)
    return hcat(fill(
        build_tspan(time(env), env.dt, env.integration_steps * mpc.horizon),
        mpc.shots)...)
end

function (mpc::RandomShooting)(env::WaveEnv)
    s = gpu(fill(state(env), mpc.shots))
    a = build_action_sequence(mpc.policy, env, mpc.horizon, mpc.shots)
    t = build_tspan(mpc, env) |> gpu

    energy = compute_energy_cost(mpc.model, s, a, t)
    penalty = compute_action_cost(a)
    cost = energy .+ mpc.alpha * penalty
    idx = argmin(cost)
    return a[1, idx]
end

random_ep1 = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode495.bson")
random_ep2 = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode496.bson")
random_ep3 = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode497.bson")
random_ep4 = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode498.bson")
random_ep5 = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode499.bson")
random_ep6 = Episode(path = "/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/episodes/episode500.bson")
_, _, t, y1 = get_energy_data(random_ep1, length(random_ep1), 1)
_, _, _, y2 = get_energy_data(random_ep2, length(random_ep2), 1)
_, _, _, y3 = get_energy_data(random_ep3, length(random_ep3), 1)
_, _, _, y4 = get_energy_data(random_ep4, length(random_ep4), 1)
_, _, _, y5 = get_energy_data(random_ep5, length(random_ep5), 1)
_, _, _, y6 = get_energy_data(random_ep6, length(random_ep6), 1)
y_random = (y1 .+ y2 .+ y3 .+ y4 .+ y5 .+ y6) ./ 6

env = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/env.bson")[:env])
node_model = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/rebuttal_node/checkpoint_step=2300/checkpoint.bson")[:model])

policy = RandomDesignPolicy(action_space(env))
horizon = 100
shots = 32
alpha = 1.0
mpc = RandomShooting(policy, node_model, horizon, shots, alpha)

reset!(env)

@time mpc(env)
# running mpc
node_mpc_ep1 = generate_episode!(mpc, env)
save(node_mpc_ep1, "node_mpc_h=$(horizon)_shots=$(shots)_ep1.bson")
node_mpc_ep2 = generate_episode!(mpc, env)
save(node_mpc_ep2, "node_mpc_h=$(horizon)_shots=$(shots)_ep2.bson")
node_mpc_ep3 = generate_episode!(mpc, env)
save(node_mpc_ep3, "node_mpc_h=$(horizon)_shots=$(shots)_ep3.bson")
node_mpc_ep4 = generate_episode!(mpc, env)
save(node_mpc_ep4, "node_mpc_h=$(horizon)_shots=$(shots)_ep4.bson")
node_mpc_ep5 = generate_episode!(mpc, env)
save(node_mpc_ep5, "node_mpc_h=$(horizon)_shots=$(shots)_ep5.bson")
node_mpc_ep6 = generate_episode!(mpc, env)
save(node_mpc_ep6, "node_mpc_h=$(horizon)_shots=$(shots)_ep6.bson")

# ## loading pml mpc results
# pml_mpc_ep1 = Episode(path = "rebuttal/pml_mpc_h=5_ep1.bson")
# pml_mpc_ep2 = Episode(path = "rebuttal/pml_mpc_h=5_ep2.bson")
# pml_mpc_ep3 = Episode(path = "rebuttal/pml_mpc_h=5_ep3.bson")
# pml_mpc_ep4 = Episode(path = "rebuttal/pml_mpc_h=5_ep4.bson")
# pml_mpc_ep5 = Episode(path = "rebuttal/pml_mpc_h=5_ep5.bson")
# pml_mpc_ep6 = Episode(path = "rebuttal/pml_mpc_h=5_ep6.bson")
# _, _, _, y1 = get_energy_data(pml_mpc_ep1, env.actions, 1)
# _, _, _, y2 = get_energy_data(pml_mpc_ep2, env.actions, 1)
# _, _, _, y3 = get_energy_data(pml_mpc_ep3, env.actions, 1)
# _, _, _, y4 = get_energy_data(pml_mpc_ep4, env.actions, 1)
# _, _, _, y5 = get_energy_data(pml_mpc_ep5, env.actions, 1)
# _, _, _, y6 = get_energy_data(pml_mpc_ep6, env.actions, 1)
# y_pml_mpc = (y1 .+ y2 .+ y3 .+ y4 .+ y5 .+ y6) ./ 6

# ## loading NODE mpc results
# node_mpc_ep1 = Episode(path = "node_mpc_ep1.bson")
# node_mpc_ep2 = Episode(path = "node_mpc_ep2.bson")
# node_mpc_ep3 = Episode(path = "node_mpc_ep3.bson")
# node_mpc_ep4 = Episode(path = "node_mpc_ep4.bson")
# node_mpc_ep5 = Episode(path = "node_mpc_ep5.bson")
# node_mpc_ep6 = Episode(path = "node_mpc_ep6.bson")
# _, _, _, y1 = get_energy_data(node_mpc_ep1, env.actions, 1)
# _, _, _, y2 = get_energy_data(node_mpc_ep2, env.actions, 1)
# _, _, _, y3 = get_energy_data(node_mpc_ep3, env.actions, 1)
# _, _, _, y4 = get_energy_data(node_mpc_ep4, env.actions, 1)
# _, _, _, y5 = get_energy_data(node_mpc_ep5, env.actions, 1)
# _, _, _, y6 = get_energy_data(node_mpc_ep6, env.actions, 1)
# y_node_mpc = (y1 .+ y2 .+ y3 .+ y4 .+ y5 .+ y6) ./ 6

# ## plotting comparison
# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Reduction of Scattered Energy With MPC")
# lines!(ax, t, y_random[:, 3], label = "Random Control")
# lines!(ax, t, y_node_mpc[:, 3], label = "NODE MPC")
# lines!(ax, t, y_pml_mpc[:, 3], label = "Ours (PML) MPC")
# axislegend(ax, position = :rb)
# save("node_control.png", fig)