using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
using ReinforcementLearning
using Interpolations: linear_interpolation
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


function build_interpolator(
        policy::AbstractPolicy,
        env::WaveEnv;
        reset::Bool = true, 
        field::Symbol = :tot)

    @assert field ∈ [:tot, :inc, :sc]

    tspans = []
    interps = DesignInterpolator[]

    x = []
    σ = []

    if reset
        RLBase.reset!(env)
    end

    while !is_terminated(env)
        tspan, interp, u_tot, u_inc = cpu(env(policy(env)))

        push!(tspans, tspan)
        push!(interps, interp)

        if field == :tot
            push!(x, u_tot)
        elseif field == :inc
            push!(x, u_inc)
        elseif field == :sc
            push!(x, u_tot .- u_inc)
        end

        push!(σ, cpu(env.signal))

        println(env.time_step)
    end

    tspan = flatten_repeated_last_dim(hcat(tspans...))

    x = flatten_repeated_last_dim(cat(x..., dims = 4))
    x = linear_interpolation(tspan, Flux.unbatch(x))

    return x, interps, σ
end



dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
@time env = gpu(BSON.load(joinpath(DATA_PATH, "env.bson"))[:env])
dim = cpu(env.dim)

MODEL_PATH = "/scratch/cmpe299-fa22/tristan/data/variable_source_yaxis_x=-10.0/models/ours_balanced_field_scale/checkpoint_step=10040/checkpoint.bson"
model = gpu(BSON.load(MODEL_PATH)[:model])
policy = RandomDesignPolicy(action_space(env))


horizon = 5 #10
shots = 256
alpha = 1.0
mpc = RandomShooting(policy, model, horizon, shots, alpha)

env.actions = 100

reset!(env)
shape = env.source.shape
x_mpc, interps_mpc, σ_mpc = build_interpolator(mpc, env, reset = false, field = :sc)

reset!(env)
env.source.shape = shape
x_random, interps_random, σ_random = build_interpolator(policy, env, reset = false, field = :sc)

mpc_signal = flatten_repeated_last_dim(cat(transpose.(σ_mpc)..., dims = 3))
random_signal = flatten_repeated_last_dim(cat(transpose.(σ_random)..., dims = 3))


t = build_tspan(0.0f0, env.dt, env.actions * env.integration_steps)
seconds = 40.0
frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
tspan = collect(range(t[1], t[end], frames))

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Random Control (Red)", xlabel = "Space (m)", ylabel = "Space (m)")
ax2 = Axis(fig[2, 1], aspect = 1.0, title = "MPC (Green)", xlabel = "Space (m)", ylabel = "Space (m)")
ax3 = Axis(fig[1:2, 2], title = "Scattered Energy in Environment", xlabel = "Time (s)", ylabel = "Energy")
xlims!(ax3, t[1], t[end])
ylims!(ax3, 0.0, max(maximum(mpc_signal[3, :]), maximum(random_signal[3, :])) * 1.20)

record(fig, "mpc.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    println(i)
    empty!(ax1)
    heatmap!(ax1, dim.x, dim.y, x_random(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
    mesh!(ax1, Waves.multi_design_interpolation(interps_random, tspan[i]))
    empty!(ax2)
    heatmap!(ax2, dim.x, dim.y, x_mpc(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
    mesh!(ax2, Waves.multi_design_interpolation(interps_mpc, tspan[i]))

    idx = findfirst(tspan[i] .<= t)[1]
    empty!(ax3)
    lines!(ax3, t[1:idx], mpc_signal[3, 1:idx], color = :green)
    lines!(ax3, t[1:idx], random_signal[3, 1:idx], color = :red)
end


fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0, title = "Random Control", xlabel = "Space (m)", ylabel = "Space (m)")
record(fig, "actions=100_random_control.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    println(i)
    empty!(ax1)
    heatmap!(ax1, dim.x, dim.y, x_random(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
    mesh!(ax1, Waves.multi_design_interpolation(interps_random, tspan[i]))
end

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0, title = "MPC", xlabel = "Space (m)", ylabel = "Space (m)")
record(fig, "actions=100_mpc.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    println(i)
    empty!(ax1)
    heatmap!(ax1, dim.x, dim.y, x_mpc(tspan[i]) .^ 2, colormap = :ice, colorrange = (0.0, 0.2))
    mesh!(ax1, Waves.multi_design_interpolation(interps_mpc, tspan[i]))
end

fig = Figure()
ax1 = Axis(fig[1, 1], title = "Scattered Energy in Environment", xlabel = "Time (s)", ylabel = "Energy")
xlims!(ax1, t[1], t[end])
ylims!(ax1, 0.0, max(maximum(mpc_signal[3, :]), maximum(random_signal[3, :])) * 1.20)
record(fig, "actions=100_scattered_energy.mp4", axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
    idx = findfirst(tspan[i] .<= t)[1]
    empty!(ax1)
    lines!(ax1, t[1:idx], mpc_signal[3, 1:idx], color = :green)
    lines!(ax1, t[1:idx], random_signal[3, 1:idx], color = :red)
end

fig = Figure()
ax1 = Axis(fig[1, 1], title = "Scattered Energy in Environment", xlabel = "Time (s)", ylabel = "Energy")
xlims!(ax1, t[1], t[end])
ylims!(ax1, 0.0, max(maximum(mpc_signal[3, :]), maximum(random_signal[3, :])) * 1.20)
empty!(ax1)
lines!(ax1, vec(t), mpc_signal[3, :], color = :green, label = "MPC")
lines!(ax1, vec(t), random_signal[3, :], color = :red, label = "Random Control")
axislegend(ax1)
save("actions=100_scattered_energy.png", fig)




# mpc_signal = render!(mpc, env, path = "mpc.mp4", energy = true, bound = 0.2f0, reset = false, field = :sc)
# mpc_signal = flatten_repeated_last_dim(cat(transpose.(mpc_signal)..., dims = 3))
# reset!(env)
# env.source.shape = shape
# random_signal = render!(policy, env, path = "random.mp4", energy = true, bound = 0.2f0, reset = false, field = :sc)
# random_signal = flatten_repeated_last_dim(cat(transpose.(random_signal)..., dims = 3))

# tspan = build_tspan(0.0f0, env.dt, size(mpc_signal, 2)-1)

# fig = Figure()
# ax = Axis(fig[1, 1], title = "Scattered Energy In Response to Actuation (50 actions)", xlabel = "Time (s)", ylabel = "Scattered Energy")
# lines!(ax, tspan, mpc_signal[3, :], label = "MPC", color = :green)
# lines!(ax, tspan, random_signal[3, :], label = "Random", color = :red)
# axislegend(ax, position = :rb)
# save("signals.png", fig)














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