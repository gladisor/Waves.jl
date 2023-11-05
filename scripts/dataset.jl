
function prepare_reconstruction_data(ep::Episode{S, Matrix{Float32}}, horizon::Int) where S
    s = S[]
    a = Vector{<: AbstractDesign}[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]
    w = Array{Float32, 3}[]
    
    n = horizon - 1
    for i in 1:(length(ep) - n - 1)
        boundary = i + n
        push!(s, ep.s[i])
        push!(a, ep.a[i:boundary])

        tspan = flatten_repeated_last_dim(hcat(ep.t[i:boundary]...))
        push!(t, tspan)

        signal = cat(ep.y[i:boundary]..., dims = 3)
        signal = permutedims(flatten_repeated_last_dim(permutedims(signal, (2, 1, 3))))
        push!(y, signal)

        push!(w, cat([ep.s[j].wave for j in i+1:boundary+1]..., dims = 3))
    end

    return s, a, t, y, w
end

function prepare_reconstruction_data(args...)
    return vcat.(prepare_reconstruction_data.(args...)...)
end

function get_reconstruction_data(ep::Episode, horizon::Int, idx::Int)
    boundary = idx + horizon - 1
    s = ep.s[idx]
    a = ep.a[idx:boundary]
    t = flatten_repeated_last_dim(hcat(ep.t[idx:boundary]...))
    y = cat(ep.y[idx:boundary]..., dims = 3)
    y = permutedims(flatten_repeated_last_dim(permutedims(y, (2, 1, 3))))
    w = cat([ep.s[j].wave for j in idx+1:boundary+1]..., dims = 3)
    return (s, a, t, y, w)
end

function get_energy_data(ep::Episode, horizon::Int, idx::Int)
    boundary = idx + horizon - 1
    s = ep.s[idx]
    a = ep.a[idx:boundary]
    t = flatten_repeated_last_dim(hcat(ep.t[idx:boundary]...))
    y = cat(ep.y[idx:boundary]..., dims = 3)
    y = permutedims(flatten_repeated_last_dim(permutedims(y, (2, 1, 3))))
    return (s, a, t, y)
end

function get_rnn_energy_data(ep::Episode, horizon::Int, idx::Int)
    boundary = idx + horizon - 1
    s = ep.s[idx:boundary]
    a = ep.a[idx:boundary]
    t = flatten_repeated_last_dim(hcat(ep.t[idx:boundary]...))
    y = cat(ep.y[idx:boundary]..., dims = 3)
    y = permutedims(flatten_repeated_last_dim(permutedims(y, (2, 1, 3))))
    return (s, a, t, y)
end

struct EpisodeDataset
    episodes::Vector{Episode}
    horizon::Int
    data_func::Function
end

function Base.length(dataset::EpisodeDataset)
    return sum(length.(dataset.episodes) .- dataset.horizon)
end

function Base.getindex(dataset::EpisodeDataset, idx::Int)
    episode_length = length(dataset.episodes[1]) - dataset.horizon
    episode_idx = (idx - 1) ÷ episode_length + 1
    data_idx = (idx - 1) % episode_length + 1
    # return get_reconstruction_data(dataset.episodes[episode_idx], horizon, data_idx)
    # return get_energy_data(dataset.episodes[episode_idx], dataset.horizon, data_idx)
    return dataset.data_func(dataset.episodes[episode_idx], dataset.horizon, data_idx)
end

function Base.getindex(dataset::EpisodeDataset, idxs::AbstractVector)
    Flux.batch.(zip([getindex(dataset, idx) for idx in idxs]...))
end