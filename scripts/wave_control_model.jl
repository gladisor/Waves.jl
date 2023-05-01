struct EpisodeData
    states::Vector{WaveEnvState}
    actions::Vector{<: AbstractDesign}
    tspans::Vector{Vector{Float32}}
    sigmas::Vector{Vector{Float32}}
end

Flux.@functor EpisodeData

Base.length(episode::EpisodeData) = length(episode.states)

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv)
    states = WaveEnvState[]
    actions = AbstractDesign[]
    tspans = []
    sigmas = []

    reset!(env)

    while !is_terminated(env)
        action = gpu(policy(env))
        tspan = build_tspan(time(env), env.dt, env.integration_steps)

        push!(states, cpu(state(env)))
        push!(actions, cpu(action))
        push!(tspans, tspan)

        env(action)

        push!(sigmas, cpu(env.σ))
    end

    return EpisodeData(states, actions, tspans, sigmas)
end

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv, episodes::Int)
    data = EpisodeData[]

    for episode in 1:episodes
        episode_data = generate_episode_data(policy, env)
        push!(data, episode_data)
    end

    return data
end

function prepare_data(episode::EpisodeData, horizon::Int)
    states = WaveEnvState[]
    actions = Vector{<:AbstractDesign}[]
    sigmas = AbstractMatrix{Float32}[]

    n = horizon - 1

    for i in 1:length(episode)
        boundary = min(i+n, length(episode))
        
        push!(states, episode.states[i])
        push!(actions, episode.actions[i:boundary])
        push!(sigmas, hcat(episode.sigmas[i:boundary]...))
    end

    return (states, actions, sigmas)
end

function prepare_data(data::Vector{EpisodeData}, horizon::Int)
    return vcat.(prepare_data.(data, horizon)...)
end

function FileIO.save(episode::EpisodeData, path::String)
    BSON.bson(path, 
        states = episode.states,
        actions = episode.actions,
        sigmas = episode.sigmas,
        tspans = episode.tspans
    )
end

function EpisodeData(;path::String)
    file = BSON.load(path)
    states = identity.(file[:states])
    actions = identity.(file[:actions])
    tspans = file[:tspans]
    sigmas = file[:sigmas]
    return EpisodeData(states, actions, tspans, sigmas)
end

struct WaveControlModel <: AbstractWaveControlModel
    wave_encoder::Chain
    design_encoder::Chain
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function (model::WaveControlModel)(h::Tuple{AbstractMatrix{Float32}, AbstractDesign}, action::AbstractDesign)
    z_wave, design = h
    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    z = model.iter(hcat(z_wave, z_design))
    sigma = model.mlp(z)
    return (z[:, [1, 2], end], design + action), sigma
end

function (model::WaveControlModel)(s::WaveEnvState, actions::Vector{<:AbstractDesign})
    z_wave = model.wave_encoder(s.wave_total)
    recur = Recur(model, (z_wave, s.design))
    return hcat([recur(action) for action in actions]...)
end

function (model::WaveControlModel)(s::WaveEnvState, action::AbstractDesign)
    return vec(model(s, [action]))
end

function encode(model::WaveControlModel, s::WaveEnvState, action::AbstractDesign)
    z_wave = model.wave_encoder(s.wave_total)
    z_design = model.design_encoder(vcat(vec(s.design), vec(action)))
    return hcat(z_wave, z_design)
end

function train(model::WaveControlModel, train_loader::DataLoader, epochs::Int, lr)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), model)

    for i in 1:epochs

        train_loss = 0.0f0
        @showprogress for (s, a, σ) in train_loader
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

function build_wave_control_model(;
        in_channels,
        h_channels,
        design_size,
        action_size,
        h_size, 
        latent_grid_size,
        latent_elements,
        latent_pml_width, 
        latent_pml_scale, 
        ambient_speed,
        dt,
        steps,
        n_mlp_layers,
        )

    wave_encoder = Chain(
        WaveEncoder(in_channels, h_channels, 2, tanh),
        Dense(4096, latent_elements, tanh),
        z -> hcat(z[:, 1], z[:, 2] * 0.0f0)
        )

    design_encoder = Chain(
        Dense(design_size + action_size, h_size, relu),
        Dense(h_size, 2 * latent_elements),
        z -> reshape(z, latent_elements, :),
        z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2]))
        )

    latent_dim = OneDim(latent_grid_size, latent_elements)
    grad = build_gradient(latent_dim)
    pml = build_pml(latent_dim, latent_pml_width, latent_pml_scale)
    bc = dirichlet(latent_dim)

    latent_dynamics = ForceLatentDynamics(ambient_speed, 1.0f0, grad, pml, bc)
    iter = Integrator(runge_kutta, latent_dynamics, 0.0f0, dt, steps)

    mlp = Chain(
        flatten,
        Dense(latent_elements * 4, h_size, relu), 
        [Dense(h_size, h_size, relu) for _ in 1:n_mlp_layers]...,
        Dense(h_size, 1), 
        vec)

    model = WaveControlModel(wave_encoder, design_encoder, iter, mlp)
    return model
end