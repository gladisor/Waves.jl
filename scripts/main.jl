using Flux
using Waves
using Waves: AbstractDesign
using CairoMakie
using ReinforcementLearning

using IntervalSets: ClosedInterval

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
    push!(hook.sols, env.sol)
    push!(hook.designs, DesignTrajectory(env))
end

mutable struct RandomDesignPolicy <: AbstractPolicy
    action::ClosedInterval{<: AbstractDesign}
end

function (policy::RandomDesignPolicy)(env::WaveEnv)
    return rand(policy.action)
end

dx = 0.1f0
ambient_speed = 1.0f0
dt = Waves.stable_dt(dx, ambient_speed)
dim = TwoDim(15.0f0, dx)
config = Scatterers(M = 4, r = 0.5f0, disk_r = 10.0f0, c = 0.1f0)

kwargs = Dict(:dim => dim, :pml_width => 4.0f0, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)

env = WaveEnv(
    initial_condition = Pulse([-9.0f0, 9.0f0], 1.0f0),
    dyn = WaveDynamics(design = config; kwargs...), 
    design_space = Waves.design_space(config, 0.5f0),
    design_steps = 5, tmax = 20.0f0)

policy = RandomDesignPolicy(action_space(env))

traj = SaveData()
@time run(policy, env, StopWhenDone(), traj)

sol = vcat(traj.sols...)
design = vcat(traj.designs...)
render!(sol, design, path = "vid.mp4")