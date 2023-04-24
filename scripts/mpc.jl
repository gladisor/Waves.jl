include("dependencies.jl")

build_cost(model::WaveMPC, s::ScatteredWaveEnvState, a::AbstractDesign) = mean(model(s, a))

function optimize_action(model::WaveMPC, s::ScatteredWaveEnvState, a::AbstractDesign)
    opt_state = Optimisers.setup(Optimisers.Adam(0.1),  a)

    for i in 1:5
        z_wave = model.wave_encoder(s.wave_total)

        cost, back = pullback(_a -> build_cost(model, s, _a) + norm(vec(_a)), a)
        gs = back(one(cost))[1]
        opt_state, a = Optimisers.update(opt_state, a, gs)
    end

    return a
end

struct MPCPolicy <: AbstractPolicy
    random::RandomDesignPolicy
    model::WaveMPC
end

function (policy::MPCPolicy)(env::ScatteredWaveEnv)
    s = gpu(state(env))
    a = gpu(policy.random(env))
    a = optimize_action(policy.model, s, a)
    return a
end

function (model::WaveMPC)(z_wave::AbstractMatrix{Float32}, da) #design::AbstractDesign, action::AbstractDesign)

    design, action = da
    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    zi = hcat(z_wave, z_design)
    z = model.iter(zi)
    z_wave = z[:, [1, 2], end]
    zf = z[:, :, end]
    return z_wave, zf
end

# function (model::WaveMPC)(s::ScatteredWaveEnvState, actions::Vector{<: AbstractDesign})

#     z_u, z_v = model.wave_encoder(s.wave_total)
#     design = s.design
#     for (i, a) in enumerate(actions)
#         z_f, z_c = model.design_encoder(vcat(vec(s.design), vec(a)))
#         zi = hcat(z_u, z_v, z_f, z_c)
#         z = model.iter(zi)
#         sigma = model.mlp(z)
#         z_u, z_v = z[:, 1, end], z[:, 2, end]
#         design = design + a
#         display(design)
#     end
# end

Flux.trainable(config::Scatterers) = (;config.r)

path = "results/radii/WaveMPC"

# model = BSON.load(joinpath(path, "model.bson"))[:model] |> gpu
env = BSON.load(joinpath(path, "env.bson"))[:env] |> gpu

policy = RandomDesignPolicy(action_space(env))
# mpc = MPCPolicy(policy, model)
;
reset!(env)

s = state(env)
a = gpu(policy(env))

design = s.design
design_size = length(vec(design))

h_size = 128
latent_elements = 512
latent_dim = OneDim(grid_size, latent_elements)
latent_dynamics = ForceLatentDynamics(ambient_speed, build_gradient(latent_dim), dirichlet(latent_dim))

wave_encoder = Chain(
    WaveEncoder(6, 8, 2, tanh), 
    Dense(1024, latent_elements, tanh),
    z -> hcat(z[:, 1], z[:, 2] * 0.0f0))

design_encoder = Chain(
    Dense(2 * design_size, h_size, relu), 
    Dense(h_size, 2 * latent_elements),
    z -> reshape(z, latent_elements, :),
    z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2]))
    )

iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)

mlp = Chain(
    flatten,
    Dense(latent_elements * 4, h_size, relu), 
    Dense(h_size, 1), 
    vec)

model = gpu(WaveMPC(wave_encoder, design_encoder, iter, mlp))

z_wave = model.wave_encoder(s.wave_total)


# model(s, [a, a, a])
# z_u, z_v = model.wave_encoder(s.wave_total)

# z_f, z_c = model.design_encoder(vcat(vec(s.design), vec(a)))
# zi = hcat(z_u, z_v, z_f, z_c)
# z = model.iter(zi)


# z_u, z_v = unbatch(z[:, [1, 2], end])

# cost = 0.0f0
# while !is_terminated(env)
#     env(mpc(env))
#     cost += reward(env)
# end
# println("MPC cost: $cost")

# reset!(env)
# cost = 0.0f0
# while !is_terminated(env)
#     env(policy(env))
#     cost += reward(env)
# end
# println("Random cost: $cost")



# # while !is_terminated(env)
# #     plot_action_distribution!(model, policy, env, path = joinpath(path, "d$(env.time_step).png"))
# #     env(policy(env))
# # end

# # validation_episode = generate_episode_data(policy, env, 10)
# # for (i, ep) in enumerate(validation_episode)
# #     plot_sigma!(model, ep, path = joinpath(path, "validation_ep$i.png"))
# # end