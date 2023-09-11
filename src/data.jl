export EpisodeData, generate_episode_data, prepare_data

struct EpisodeData
    states::Vector{WaveEnvState}
    actions::Vector{<: AbstractDesign}
    tspans::Vector{Vector{Float32}}
    signals::Vector{Vector{Float32}}
end

Base.length(episode::EpisodeData) = length(episode.states)

function generate_episode_data(policy::AbstractPolicy, env::WaveEnv)
    states = WaveEnvState[]
    actions = AbstractDesign[]
    tspans = []
    signals = []

    reset!(env)

    while !is_terminated(env)
        action = policy(env)
        tspan = build_tspan(time(env), env.dt, env.integration_steps)

        push!(states, cpu(state(env)))
        push!(actions, cpu(action))
        push!(tspans, tspan)

        env(action)

        push!(signals, cpu(env.signal))
    end

    return EpisodeData(states, actions, tspans, signals)
end

function prepare_data(episode::EpisodeData, horizon::Int)
    states = WaveEnvState[]
    actions = Vector{<: AbstractDesign}[]
    tspans = AbstractMatrix{Float32}[]

    signals = eltype(episode.signals)[]
    signal_dim = ndims(first(episode.signals))

    n = horizon - 1
    for i in 1:(length(episode) - n)
        boundary = i + n
        
        push!(states, episode.states[i])
        push!(actions, episode.actions[i:boundary])
        push!(tspans, hcat(episode.tspans[i:boundary]...))
        push!(signals, flatten_repeated_last_dim(cat(episode.signals[i:boundary]..., dims = signal_dim + 1)))
    end

    return (states, actions, tspans, signals)
end

function prepare_data(data::Vector{EpisodeData}, horizon::Int)
    return vcat.(prepare_data.(data, horizon)...)
end

function FileIO.save(episode::EpisodeData, path::String)
    BSON.bson(path, 
        states = episode.states,
        actions = episode.actions,
        tspans = episode.tspans,
        signals = episode.signals,
    )
end

function EpisodeData(;path::String)
    file = BSON.load(path)
    states = identity.(file[:states])
    actions = identity.(file[:actions])
    tspans = file[:tspans]
    signals = file[:signals]
    return EpisodeData(states, actions, tspans, signals)
end