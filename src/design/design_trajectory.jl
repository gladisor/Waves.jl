export DesignTrajectory

struct DesignTrajectory{D <: AbstractDesign}
    traj::Vector{D}
end

function DesignTrajectory(dyn::WaveDynamics, n::Int)
    design = dyn.design
    t = collect(range(design.ti, design.tf, n + 1))
    traj = typeof(design.initial)[]

    for i ∈ axes(t, 1)
        push!(traj, design(t[i]))
    end

    return DesignTrajectory(traj)
end

function DesignTrajectory(env::WaveEnv)
    return DesignTrajectory(env.dyn, env.design_steps)
end

function Base.vcat(dt1::DesignTrajectory, dt2::DesignTrajectory)
    pop!(dt1.traj)
    return DesignTrajectory(vcat(dt1.traj, dt2.traj))
end

function Base.vcat(dts::DesignTrajectory...)
    return reduce(vcat, dts)
end

function Base.length(dt::DesignTrajectory)
    return length(dt.traj)
end

function Base.getindex(dt::DesignTrajectory, i::Int64)
    return dt.traj[i]
end

function Flux.cpu(dt::DesignTrajectory)
    return DesignTrajectory(cpu.(dt.traj))
end