export WaveSol

"""
Structure for storing the minimal amount of information needed to represent the solution
of a wave simulation
"""
struct WaveSol{D <: AbstractDim}
    dim::D
    t::Vector{<: AbstractFloat}
    u::Vector{<: AbstractArray{<: AbstractFloat}}
end

function Base.length(sol::WaveSol)
    return length(sol.t)
end

"""
Subtracts one wave solution from another. Maintains the first solution's time and dim
"""
function Base.:-(sol1::WaveSol, sol2::WaveSol)
    u = [u1 .- u2 for (u1, u2) in zip(sol1.u, sol2.u)]
    return WaveSol(sol1.dim, sol1.t, u)
end

function Base.getindex(sol::WaveSol, i::Int64)
    return sol.u[i]
end

function Base.lastindex(sol::WaveSol)
    return sol.u[end]
end