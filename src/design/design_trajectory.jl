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