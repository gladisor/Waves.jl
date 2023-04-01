export DesignTrajectory

struct DesignTrajectory{D <: AbstractDesign}
    traj::Vector{D}
end

function DesignTrajectory(dts::DesignTrajectory{D}...) where D <: AbstractDesign
    traj = D[]

    for design_trajectory ∈ dts
        for i ∈ 1:(length(design_trajectory) - 1)
            push!(traj, design_trajectory[i])
        end
    end

    push!(traj, dts[end].traj[end])

    return DesignTrajectory(traj)
end

function DesignTrajectory(design::DesignInterpolator, n::Int)

    t = collect(range(design.ti, design.tf, n + 1))
    traj = typeof(design.initial)[]

    for i ∈ axes(t, 1)
        push!(traj, design(t[i]))
    end

    return DesignTrajectory(traj)
end

function DesignTrajectory(states::Vector{WaveEnvState}, actions::Vector{ <: AbstractDesign})
    designs = DesignTrajectory[]

    for (s, a) ∈ zip(states, actions)
        interp = DesignInterpolator(s.design, a, s.sol.total.t[1], s.sol.total.t[end])
        dt = DesignTrajectory(interp, length(s.sol.total)-1)
        push!(designs, dt)
    end

    return DesignTrajectory(designs...)
end

function Base.length(dt::DesignTrajectory)
    return length(dt.traj)
end

function Base.getindex(dt::DesignTrajectory, i::Int64)
    return dt.traj[i]
end

function Flux.gpu(dt::DesignTrajectory)
    return DesignTrajectory(gpu.(dt.traj))
end

function Flux.cpu(dt::DesignTrajectory)
    return DesignTrajectory(cpu.(dt.traj))
end