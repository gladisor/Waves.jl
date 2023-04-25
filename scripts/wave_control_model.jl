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

        @time env(action)

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

# function (model::WaveMPC)(z_wave::AbstractMatrix{Float32}, design::AbstractDesign, action::AbstractDesign)
#     z_design = model.design_encoder(vcat(vec(design), vec(action)))
#     zi = hcat(z_wave, z_design)
#     z = model.iter(zi)
#     z_wave = z[:, [1, 2], end]
#     sigma = model.mlp(z)
#     return z_wave, sigma
# end

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
    return hcat([recur(a) for a in actions]...)
end

# function (model::WaveMPC)(s::ScatteredWaveEnvState, action::AbstractDesign)
#     z_wave = model.wave_encoder(s.wave_total)
#     _, sigma = model(z_wave, s.design, action)
#     return sigma
# end

# function build_cost(model::WaveMPC, s::ScatteredWaveEnvState, actions::Vector{AbstractDesign})
#     z_wave = model.wave_encoder(s.wave_total)
#     design = s.design

#     cost = 0.0f0

#     for a in actions
#         z_wave, sigma = model(z_wave, design, a)
#         cost += mean(sigma)
#         design += a
#     end

#     return cost
# end

# function build_loss(model::WaveMPC, s::ScatteredWaveEnvState, actions::Vector{<: AbstractDesign}, sigma::AbstractMatrix{Float32})
#     z_wave = model.wave_encoder(s.wave_total)
#     design = s.design

#     loss = 0.0f0

#     for (i, a) in enumerate(actions)
#         z_wave, sigma_pred = model(z_wave, design, a)
#         loss += mse(sigma_pred, sigma[:, i])
#         design += a
#     end

#     return loss
# end