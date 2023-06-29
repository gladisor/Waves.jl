export EpisodeData, generate_episode_data, prepare_data

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
        action = policy(env)
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
    tspans = AbstractMatrix{Float32}[]
    sigmas = AbstractMatrix{Float32}[]

    n = horizon - 1

    for i in 1:(length(episode) - n)
        boundary = i + n
        
        push!(states, episode.states[i])
        push!(actions, episode.actions[i:boundary])
        push!(tspans, hcat(episode.tspans[i:boundary]...))
        push!(sigmas, hcat(episode.sigmas[i:boundary]...))
    end

    return (states, actions, tspans, sigmas)
end

function prepare_data(data::Vector{EpisodeData}, horizon::Int)
    return vcat.(prepare_data.(data, horizon)...)
end

function FileIO.save(episode::EpisodeData, path::String)
    BSON.bson(path, 
        states = episode.states,
        actions = episode.actions,
        tspans = episode.tspans,
        sigmas = episode.sigmas,
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


get_step_number(path::String) = parse(Int, split(split(path, "step_")[end], ".bson")[1])

function convert_to_observations(episode_paths::Vector{String}, horizon::Int)
    sorted_episode_paths = sort(episode_paths, by = get_step_number)

    observations = Vector{String}[]

    for i in 1:(length(sorted_episode_paths) - horizon + 1)

        push!(
            observations, 
            sorted_episode_paths[i:(i + horizon - 1)]
            )
    end

    return observations
end

function convert_to_datapoint(steps::Vector{String})
    steps = BSON.load.(steps)

    s = steps[1][:state]
    a = [step[:action] for step in steps]
    t = hcat([step[:tspan] for step in steps]...)
    sigma = hcat([step[:sigma] for step in steps]...)

    return (s, a, t, sigma)
end

struct EpisodeDataset
    observations::Vector{Vector{String}}
end

function EpisodeDataset(path::String, horizon::Int)
    episode_paths = readdir.(readdir(path, join = true), join = true)
    observations = vcat(convert_to_observations.(episode_paths, horizon)...)
    return EpisodeDataset(observations)
end

function Flux.numobs(data::EpisodeDataset)
    return length(data.observations)
end

function Flux.getobs(data::EpisodeDataset, idx::Int)
    return convert_to_datapoint(data.observations[idx])
end

function Flux.getobs(data::EpisodeDataset, idx::Vector{Int})
    states, actions, tspans, sigmas = zip(Flux.getobs.([data], idx)...)
    return ([states...], [actions...], [tspans...], [sigmas...])
end