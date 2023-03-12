export SaveData, DesignStates

mutable struct SaveData <: AbstractHook
    sols::Vector{TotalWaveSol}
    designs::Vector{DesignTrajectory}
end

function SaveData()
    return SaveData(TotalWaveSol[], DesignTrajectory[])
end

function (hook::SaveData)(::PreEpisodeStage, agent, env::WaveEnv)
    hook.sols = TotalWaveSol[]
    hook.designs = DesignTrajectory[]
end

function (hook::SaveData)(::PostActStage, agent, env::WaveEnv)
    push!(hook.sols, cpu(env.sol))
    push!(hook.designs, cpu(DesignTrajectory(env)))
end

mutable struct DesignStates <: AbstractHook
    states::Vector{<: AbstractDesign}
end

function DesignStates()
    return DesignStates(AbstractDesign[])
end

function (hook::DesignStates)(::PreActStage, agent, env::WaveEnv, action)
    push!(hook.states, env.total_dynamics.design(time(env)))
end