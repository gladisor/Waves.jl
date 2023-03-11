export WaveSol, TotalWaveSol

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

function Base.getindex(sol::WaveSol, i)
    return sol.u[i]
end

function Flux.gpu(sol::WaveSol)
    return WaveSol(gpu(sol.dim), gpu(sol.t), gpu(sol.u))
end

function Flux.cpu(sol::WaveSol)
    return WaveSol(cpu(sol.dim), cpu(sol.t), cpu(sol.u))
end

function WaveSol(sols::WaveSol...)
    t = Float32[]
    u = typeof(first(sols).u[1])[]

    for sol ∈ sols
        for i ∈ 1:(length(sol.u) - 1)
            push!(t, sol.t[i])
            push!(u, sol.u[i])
        end
    end

    push!(t, last(sols).t[end])
    push!(u, last(sols).u[end])

    return WaveSol(first(sols).dim, t, u)
end

struct TotalWaveSol
    total::WaveSol
    incident::WaveSol
    scattered::WaveSol
end

function TotalWaveSol(;total::WaveSol, incident::WaveSol)
    return TotalWaveSol(total, incident, total - incident)
end

function TotalWaveSol(sols::TotalWaveSol...)
    total = WaveSol[]
    incident = WaveSol[]
    scattered = WaveSol[]

    for sol ∈ sols
        push!(total, sol.total)
        push!(incident, sol.incident)
        push!(scattered, sol.scattered)
    end

    return TotalWaveSol(WaveSol(total...), WaveSol(incident...), WaveSol(scattered...))
end

function Flux.gpu(sol::TotalWaveSol)
    return TotalWaveSol(gpu(sol.total), gpu(sol.incident), gpu(sol.scattered))
end

function Flux.cpu(sol::TotalWaveSol)
    return TotalWaveSol(cpu(sol.total), cpu(sol.incident), cpu(sol.scattered))
end