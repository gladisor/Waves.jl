export save_wave_solutions!, load_wave_solutions, time_data

function save_wave_solutions!(data::Vector{WaveSolution1D}, path::String)
    BSON.@save path data
end

function load_wave_solutions(path::String)
    return map(identity, BSON.load(path)[:data])
end

function time_data(x::WaveSolution1D)
    u0 = x.u[:, 1]
    u = x.u[:, 2:end]
    t = x.t[2:end]

    u0 = vcat(repeat(u0, 1, length(t)), t')

    return (u0, u)
end

function time_data(x::Vector{<: WaveSolution1D})
    x = map(time_data, x)
    u0, u = zip(x...)
    u0 = hcat(u0...)
    u = hcat(u...)

    return (u0, u)
end