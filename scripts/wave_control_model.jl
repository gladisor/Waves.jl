struct EpisodeData
    states::Vector{ScatteredWaveEnvState}
    actions::Vector{<: AbstractDesign}
    tspans::Vector{Vector{Float32}}
    sigmas::Vector{Vector{Float32}}
end

Flux.@functor EpisodeData

Base.length(episode::EpisodeData) = length(episode.states)

function generate_episode_data(policy::AbstractPolicy, env::ScatteredWaveEnv)
    states = ScatteredWaveEnvState[]
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

function generate_episode_data(policy::AbstractPolicy, env::ScatteredWaveEnv, episodes::Int)
    data = EpisodeData[]

    for episode in 1:episodes
        episode_data = generate_episode_data(policy, env)
        push!(data, episode_data)
    end

    return data
end

function prepare_data(episode::EpisodeData, n::Int)
    states = ScatteredWaveEnvState[]
    actions = Vector{<:AbstractDesign}[]
    sigmas = AbstractMatrix{Float32}[]

    for i in 1:(length(episode) - n)
        push!(states, episode.states[i])
        push!(actions, episode.actions[i:i+n])
        push!(sigmas, hcat(episode.sigmas[i:i+n]...))
    end

    return (states, actions, sigmas)
end

function prepare_data(data::Vector{EpisodeData}, n::Int)
    vcat.(prepare_data.(data, n)...)
end

struct WaveMPC <: AbstractWaveControlModel
    wave_encoder::Chain
    design_encoder::Chain
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveMPC

function (model::WaveMPC)(h::Tuple{AbstractMatrix{Float32}, AbstractDesign}, action::AbstractDesign)
    z_wave, design = h
    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    z = model.iter(hcat(z_wave, z_design))
    sigma = model.mlp(z)
    return (z[:, [1, 2], end], design + action), sigma
end

function (model::WaveMPC)(s::ScatteredWaveEnvState, actions::Vector{<:AbstractDesign})
    z_wave = model.wave_encoder(s.wave_total)
    recur = Recur(model, (z_wave, s.design))
    return hcat([recur(action) for action in actions]...)
end

function (model::WaveMPC)(s::ScatteredWaveEnvState, action::AbstractDesign)
    return vec(model(s, [action]))
end

function encode(model::WaveMPC, s::ScatteredWaveEnvState, action::AbstractDesign)
    z_wave = model.wave_encoder(s.wave_total)
    z_design = model.design_encoder(vcat(vec(s.design), vec(action)))
    return hcat(z_wave, z_design)
end

function train(model::WaveMPC, train_loader::DataLoader, epochs::Int, )
    opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)

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