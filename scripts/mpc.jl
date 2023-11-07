using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using ReinforcementLearning
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(2)
display(Flux.device())
include("model_modifications.jl")

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int)
    return [policy(env) for i in 1:horizon]
end

function build_action_sequence(policy::AbstractPolicy, env::AbstractEnv, horizon::Int, shots::Int)
    return hcat([build_action_sequence(policy, env, horizon) for i in 1:shots]...)
end

struct RandomShooting <: AbstractPolicy
    policy::AbstractPolicy
    model
    horizon::Int
    shots::Int
    alpha::Float32
end

function compute_action_cost(a::Matrix{<: AbstractDesign})
    x = cat([hcat(vec.(a)[:, i]...) for i in axes(a, 2)]..., dims = 3)
    return vec(sum(sqrt.(sum(x .^ 2, dims = 1)), dims = 2))
end

function compute_energy_cost(model::AcousticEnergyModel, s, a, t)
    y_hat = model(s, a, t)
    return vec(sum(y_hat[:, 3, :], dims = 1))
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
_, _, t, y1 = prepare_data(random_ep1, length(random_ep1))
_, _, _, y2 = prepare_data(random_ep2, length(random_ep2))
_, _, _, y3 = prepare_data(random_ep3, length(random_ep3))
_, _, _, y4 = prepare_data(random_ep4, length(random_ep4))
_, _, _, y5 = prepare_data(random_ep5, length(random_ep5))
_, _, _, y6 = prepare_data(random_ep6, length(random_ep6))
y_random = (y1 .+ y2 .+ y3 .+ y4 .+ y5 .+ y6) ./ 6
y_random = y_random[1]
t = t[1]

env = gpu(BSON.load("/home/012761749/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/env.bson")[:env])
pml_checkpoint = 2300
MODEL_PATH = "/scratch/cmpe299-fa22/tristan/data/AcousticDynamics{TwoDim}_Cloak_Pulse_dt=1.0e-5_steps=100_actions=200_actionspeed=250.0_resolution=(128, 128)/trainable_pml_localization_horizon=20_batchsize=32_h_size=256_latent_gs=100.0_pml_width=10.0_nfreq=500/checkpoint_step=$pml_checkpoint/checkpoint.bson"
model = gpu(BSON.load(MODEL_PATH)[:model])

policy = RandomDesignPolicy(action_space(env))

## horizon = 100, shots = 32 works
horizon = 10
shots = 256
println("Horizon = $horizon, Shots = $shots")
alpha = 1.0
mpc = RandomShooting(policy, model, horizon, shots, alpha)

reset!(env)
mpc_ep1 = generate_episode!(mpc, env)
save(mpc_ep1, "pml_mpc_ep1_h=$(horizon)_shots=$(shots).bson")
mpc_ep2 = generate_episode!(mpc, env)
save(mpc_ep2, "pml_mpc_ep2_h=$(horizon)_shots=$(shots).bson")
mpc_ep3 = generate_episode!(mpc, env)
save(mpc_ep3, "pml_mpc_ep3_h=$(horizon)_shots=$(shots).bson")
mpc_ep4 = generate_episode!(mpc, env)
save(mpc_ep4, "pml_mpc_ep4_h=$(horizon)_shots=$(shots).bson")
mpc_ep5 = generate_episode!(mpc, env)
save(mpc_ep5, "pml_mpc_ep5_h=$(horizon)_shots=$(shots).bson")
mpc_ep6 = generate_episode!(mpc, env)
save(mpc_ep6, "pml_mpc_ep6_h=$(horizon)_shots=$(shots).bson")

# mpc_ep1 = Episode(path = "mpc_ep1.bson")
# _, _, _, y1 = prepare_data(mpc_ep1, env.actions)
# y1 = y1[1]
# _, _, _, y2 = prepare_data(mpc_ep2, env.actions)
# y2 = y2[1]
# _, _, _, y3 = prepare_data(mpc_ep3, env.actions)
# y3 = y4[1]
# _, _, _, y4 = prepare_data(mpc_ep4, env.actions)
# y4 = y4[1]
# _, _, _, y5 = prepare_data(mpc_ep5, env.actions)
# y5 = y5[1]
# _, _, _, y6 = prepare_data(mpc_ep6, env.actions)
# y6 = y6[1]
# y_mpc = (y1 .+ y2 .+ y3 .+ y4 .+ y5 .+ y6) ./ 6

# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Reduction of Scattered Energy With MPC")
# lines!(ax, t, y_random[:, 3], label = "Random Control")
# lines!(ax, t, y_mpc[:, 3], label = "MPC")
# save("mpc_comparison.png", fig)