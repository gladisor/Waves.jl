export SaveData

mutable struct SaveData <: AbstractHook
    sols::Vector{WaveSol}
    designs::Vector{DesignTrajectory}
end

function SaveData()
    return SaveData(WaveSol[], DesignTrajectory[])
end

function (hook::SaveData)(::PreEpisodeStage, agent, env::WaveEnv)
    hook.sols = WaveSol[]
    hook.designs = DesignTrajectory[]
end

function (hook::SaveData)(::PostActStage, agent, env::WaveEnv)
    push!(hook.sols, cpu(env.sol))
    push!(hook.designs, cpu(DesignTrajectory(env)))
end
