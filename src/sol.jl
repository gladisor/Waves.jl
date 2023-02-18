export WaveSol

"""
Structure for storing the minimal amount of information needed to represent the solution
of a wave simulation
"""
struct WaveSol{D <: AbstractDim}
    dim::D
    t::AbstractVector{Float32}
    u::AbstractVector{<: AbstractArray{Float32}}
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

function Base.vcat(sol1::WaveSol, sol2::WaveSol)
    pop!(sol1.t)
    pop!(sol1.u)
    return WaveSol(sol1.dim, vcat(sol1.t, sol2.t), vcat(sol1.u, sol2.u))
end

function Base.vcat(sols::WaveSol...)
    return reduce(vcat, sols)
end

function Flux.cpu(sol::WaveSol)
    return WaveSol(cpu(sol.dim), cpu(sol.t), cpu(sol.u))
end