export SaveData

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
