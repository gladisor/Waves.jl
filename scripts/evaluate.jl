include("dependencies.jl")

function compute_cost(
        sigma::AbstractMatrix{Float32}, 
        action_sequence::Vector{<:AbstractDesign},
        alpha::Float32)

    return sum(sigma[2:end, :]) + alpha * sum(norm.(vec.(action_sequence)))
end

# function random_shooting(mpc::MPC, s::WaveEnvState, shots::Int; alpha::Float32 = 500.0f0)
#     as = gpu([[mpc.policy(env) for _ in 1:mpc.horizon] for _ in 1:shots])
#     sigmas = mpc.model.([s], as)
#     costs = compute_cost.(sigmas, as, alpha)
#     idx = argmin(costs)
#     return (costs[idx], as[idx])
# end

function optimise_actions(opt_state, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign})
    cost, back = pullback(_a -> compute_cost(model(s, _a), _a, 200.0f0), a)
    gs = back(one(cost))[1]
    opt_state, a = Optimisers.update(opt_state, a, gs)
    return opt_state, a
end

function optimise_actions(opt::AbstractRule, model::WaveControlModel, s::WaveEnvState, a::Vector{<: AbstractDesign}, steps::Int)
    opt_state = Optimisers.setup(opt, a)

    for i in 1:steps
        opt_state, a = optimise_actions(opt_state, model, s, a)
    end

    return a
end

struct MPC <: AbstractPolicy
    policy::AbstractPolicy
    model::WaveControlModel
    opt::AbstractRule
    horizon::Int
    opt_steps::Int
end

Flux.@functor MPC

function (mpc::MPC)(env::WaveEnv)
    s = gpu(state(env))
    a = gpu([mpc.policy(env) for _ in 1:mpc.horizon])
    a = optimise_actions(mpc.opt, mpc.model, s, a, mpc.opt_steps)
    return a[1]
end

path = "results/M=6_core=1.0/h_channels=32_h_size=1024_latent_elements=512_n_mlp_layers=3_lr=0.0001_horizon=3_epochs=10/transfer_lr=8.0e-5_horizon=5/"
model_path = joinpath(path, "model10.bson")
env_path = "data/M=6_as=1.0/env.bson"

model = BSON.load(model_path)[:model] |> gpu
env = BSON.load(env_path)[:env] |> gpu
policy = RandomDesignPolicy(action_space(env))
reset!(env)

horizon = 5
opt_steps = 2
episodes = 5
lr = 1e-4

opt = Optimisers.Descent(lr)
mpc = gpu(MPC(policy, model, opt, horizon, opt_steps))

mpc_hook = TotalRewardPerEpisode()
run(mpc, env, StopAfterEpisode(episodes), mpc_hook)

random_hook = TotalRewardPerEpisode()
run(policy, env, StopAfterEpisode(episodes), random_hook)

avg_mpc = mean(mpc_hook.rewards)
avg_random = mean(random_hook.rewards)

println("MPC: $avg_mpc")
println("RANDOM: $avg_random")

@time render!(mpc, env, path = joinpath(path, "mpc.mp4"), seconds = 50.0f0)
@time render!(policy, env, path = joinpath(path, "random.mp4"), seconds = 50.0f0)