export time_data

function time_data(sol::WaveSol{OneDim})
    t = sol.sol.t[2:end]
    u = Waves.get_data(sol)
    u0 = u[1]
    u = hcat(u[2:end]...)
    u0 = vcat(repeat(u0, 1, length(t)), t')
    return (u0, u)
end

function time_data(x::Vector{<: WaveSol{OneDim}})
    x = map(time_data, x)
    u0, u = zip(x...)
    u0 = hcat(u0...)
    u = hcat(u...)

    return (u0, u)
end