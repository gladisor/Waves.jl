using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using ReinforcementLearning
Flux.CUDA.allowscalar(false)
println("Loaded Packages")
Flux.device!(0)
display(Flux.device())

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

function compute_energy_cost(model::WaveControlPINN, s, a, t)
    @time y_hat_1 = model(s[1:64], a[:, 1:64], t[:, 1:64])
    @time y_hat_2 = model(s[65:128], a[:, 65:128], t[:, 65:128])
    @time y_hat_3 = model(s[129:192], a[:, 129:192], t[:, 129:192])
    @time y_hat_4 = model(s[193:end], a[:, 193:end], t[:, 193:end])
    y_hat = vcat(y_hat_1, y_hat_2, y_hat_3, y_hat_4)
    return vec(sum(y_hat[:, 3, :], dims = 1))
end

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
@time env = gpu(BSON.load(joinpath(DATA_PATH, "env.bson"))[:env])

# MODEL_PATH = "/scratch/cmpe299-fa22/tristan/data/variable_source_yaxis_x=-10.0/models/ours_balanced_field_scale/checkpoint_step=10040/checkpoint.bson"
MODEL_PATH = "/scratch/cmpe299-fa22/tristan/data/variable_source_yaxis_x=-10.0/models/wave_control_pinn_accumulate=8/checkpoint_step=80320/checkpoint.bson"
model = gpu(BSON.load(MODEL_PATH)[:model])
policy = RandomDesignPolicy(action_space(env))

reset!(env)

for i in 1:5
    @time env(policy(env))
end

dim = cpu(env.dim)
fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
hidedecorations!(ax)
hidespines!(ax)  
hidexdecorations!(ax)
hideydecorations!(ax)
heatmap!(ax, dim.x, dim.y, cpu(env.wave[:, :, 1, end]), colormap = :ice, colorrange = (-0.5, 0.5))
mesh!(ax, cpu(env.design))
save("wave.png", fig)




# dim = cpu(env.dim)
# reset!(env)
# env.source.shape = build_normal(env.source.grid, env.source.μ_high, env.source.σ, env.source.a) ## build normal distribution shape
# for i in 1:15
#     @time env(policy(env))
# end
# wave_1 = cpu(env.wave[:, :, 1, end])
# design_1 = cpu(env.design)

# reset!(env)
# env.source.shape = build_normal(env.source.grid, (3 * env.source.μ_high .+ env.source.μ_low) / 4, env.source.σ, env.source.a) ## build normal distribution shape
# for i in 1:15
#     @time env(policy(env))
# end
# wave_2 = cpu(env.wave[:, :, 1, end])
# design_2 = cpu(env.design)

# reset!(env)
# env.source.shape = build_normal(env.source.grid, (env.source.μ_high .+ 3 * env.source.μ_low) / 4, env.source.σ, env.source.a) ## build normal distribution shape
# for i in 1:15
#     @time env(policy(env))
# end
# wave_3 = cpu(env.wave[:, :, 1, end])
# design_3 = cpu(env.design)

# reset!(env)
# env.source.shape = build_normal(env.source.grid, env.source.μ_low, env.source.σ, env.source.a) ## build normal distribution shape
# for i in 1:15
#     @time env(policy(env))
# end
# wave_4 = cpu(env.wave[:, :, 1, end])
# design_4 = cpu(env.design)


# fig = Figure(resolution = (600, 600))
# ax1 = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax1, dim.x, dim.y, wave_1, colormap = :ice, colorrange = (-0.5f0, 0.5f0))
# mesh!(ax1, design_1)

# # ax2 = Axis(fig[1, 2], aspect = 1.0f0)
# # heatmap!(ax2, dim.x, dim.y, wave_2, colormap = :ice, colorrange = (-1.0f0, 1.0f0))
# # mesh!(ax2, design_2)

# ax3 = Axis(fig[1, 2], aspect = 1.0f0)
# heatmap!(ax3, dim.x, dim.y, wave_3, colormap = :ice, colorrange = (-0.5f0, 0.5f0))
# mesh!(ax3, design_3)

# # ax4 = Axis(fig[2, 2], aspect = 1.0f0)
# # heatmap!(ax4, dim.x, dim.y, wave_4, colormap = :ice, colorrange = (-1.0f0, 1.0f0))
# # mesh!(ax4, design_4)
# fig.layout.default_colgap = Fixed(5.0f0)
# fig.layout.default_rowgap = Fixed(5.0f0)
# save("wave.png", fig)






# # horizon = 10
# # shots = 256
# # alpha = 1.0
# # mpc = RandomShooting(policy, model, horizon, shots, alpha)
# # y_hat = mpc(env)

# # delta_mu = (env.source.μ_high .- env.source.μ_low)
# # x = gpu(collect(range(0.0f0, 1.0f0, 5)))
# # mu = env.source.μ_low .+ delta_mu .* x

# # for location in axes(mu, 1)
# #     shape = build_normal(env.source.grid, mu[[location], :], env.source.σ, env.source.a)

# #     for episode in 1:4
# #         reset!(env)
# #         env.source.shape = shape
# #         mpc_ep = generate_episode!(mpc, env, reset = false)
# #         # save(mpc_ep, "control_results/cPILS_location=$location,episode=$episode.bson")
# #         save(mpc_ep, "control_results/PINC_location=$location,episode=$episode.bson")
# #         # reset!(env)
# #         # env.source.shape = shape
# #         # random_ep = generate_episode!(policy, env, reset = false)
# #         # save(random_ep, "control_results/random_location=$location,episode=$episode.bson")
# #     end
# # end