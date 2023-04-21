include("dependencies.jl")

env = gpu(BSON.load("results/radii/PercentageWaveControlModel/env.bson")[:env])
model = gpu(BSON.load("results/radii/PercentageWaveControlModel/model.bson")[:model])
reset!(env)

policy = RandomDesignPolicy(action_space(env))

for i in 1:3
    env(policy(env))
end

tspan = build_tspan(time(env), env.dt, env.integration_steps)

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0f0, xlabel = "Time (s)", ylabel = "Scattered Energy (σ)", title = "Effect of random actions on real dynamics")

for i in 1:4
    temp_env = deepcopy(env)
    temp_env(policy(temp_env))
    lines!(ax1, tspan, temp_env.σ)
end

ax2 = Axis(fig[1, 2], aspect = 1.0f0, xlabel = "Time (s)", ylabel = "Scattered Energy (σ)", title = "Effect of random actions on model dynamics")

s = state(env)
for i in 1:4
    sigma_pred = cpu(model(s, gpu(policy(env))))
    lines!(ax2, tspan, sigma_pred)
end

save("sigma_compare.png", fig)