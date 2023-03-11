using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)

using Waves

mutable struct DesignStates <: AbstractHook
    states::Vector{<: AbstractDesign}
end

function DesignStates()
    return DesignStates(AbstractDesign[])
end

function (hook::DesignStates)(::PreActStage, agent, env::WaveEnv, action)
    println(time(env))
    push!(hook.states, env.total_dynamics.design(time(env)))
end

dim = TwoDim(5.0f0, 128)
dynamics_kwargs = Dict(:dim => dim, :pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
cyl = Cylinder(0.0f0, 0.0f0, 0.5f0, 0.5f0)
tmax = 10.0f0

env = WaveEnv(
    initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
    wave = build_wave(dim, fields = 6),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = cyl,
    space = design_space(cyl, 1.0f0),
    design_steps = 100,
    tmax = tmax;
    dynamics_kwargs...) |> gpu

policy = RandomDesignPolicy(action_space(env))
traj = episode_trajectory(env)
agent = Agent(policy, traj)

design_states = DesignStates()
@time run(agent, env, StopWhenDone(), design_states)

wave_states = traj.traces.state[2:end]
design_states = design_states.states
a = traj.traces.action[1:end-1]