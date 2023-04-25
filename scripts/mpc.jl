include("dependencies.jl")

function train(model::WaveMPC, train_loader::DataLoader, epochs::Int)
    opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)

    for i in 1:epochs
        train_loss = 0.0f0

        for (s, a, σ) in train_loader
            s, a, σ = gpu(s[1]), gpu(a[1]), gpu(σ[1])

            loss, back = pullback(_model -> mse(_model(s, a), σ), model)
            gs = back(one(loss))[1]

            opt_state, model = Optimisers.update(opt_state, model, gs)
            train_loss += loss
        end

        print("Epoch: $i, Loss: ")
        println(train_loss / length(train_loader))
    end

    return model
end

env = BSON.load("env.bson")[:env] |> gpu
reset!(env)
model = BSON.load("model.bson")[:model] |> gpu
policy = RandomDesignPolicy(action_space(env))

val_episodes = generate_episode_data(policy, env, 10)

for (i, episode) in enumerate(val_episodes)
    plot_sigma!(model, episode, path = "val_episode$i.png")
end

# path = "results/radii/WaveMPC"
# env = BSON.load(joinpath(path, "env.bson"))[:env] |> gpu
# policy = RandomDesignPolicy(action_space(env))
# ;

# reset!(env)
# s = state(env)
# a = gpu(policy(env))

# design = s.design
# design_size = length(vec(design))

# latent_dim = OneDim(grid_size, latent_elements)
# latent_dynamics = ForceLatentDynamics(ambient_speed, build_gradient(latent_dim), dirichlet(latent_dim))

# wave_encoder = Chain(
#     WaveEncoder(6, 8, 2, tanh), 
#     Dense(1024, latent_elements, tanh),
#     z -> hcat(z[:, 1], z[:, 2] * 0.0f0))

# design_encoder = Chain(
#     Dense(2 * design_size, h_size, relu), 
#     Dense(h_size, 2 * latent_elements),
#     z -> reshape(z, latent_elements, :),
#     z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2]))
#     )

# iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)
# mlp = Chain(flatten, Dense(latent_elements * 4, h_size, relu), Dense(h_size, 1), vec)

# model = gpu(WaveMPC(wave_encoder, design_encoder, iter, mlp))

# # data = generate_episode_data(policy, env, 50)
# # train_loader = Flux.DataLoader(prepare_data(data, 1), shuffle = true)

# # model = train(model, train_loader, 20)
# # ;

# # ## saving model and env
# # model = cpu(model)
# # @save "model.bson" model
# # env = cpu(env)
# # @save "env.bson" env

# # val_episodes = generate_episode_data(policy, env, 2)

# # for (i, episode) in enumerate(val_episodes)
# #     plot_sigma!(model, episode, path = "val_episode$i.png")
# # end

# episode = val_episodes[1]
# episode.actions
# sigma = model(gpu(episode.states[1]), gpu(episode.actions))
# sigma = cpu(vec(sigma[2:end, :]))

# tspan = hcat(episode.tspans...)
# tspan = vec(tspan[2:end, :])

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, tspan, sigma)
# save("sigma.png", fig)

# states, actions, sigmas = prepare_data(episode, 1)
# sigma_pred = model(gpu(states[1]), gpu(actions[1]))

# episode.tspans[1]

# plot_episode_data!(episode, cols = 5, path = "data.png")