using ReinforcementLearning
using IntervalSets
using Flux

using Waves

function episode_trajectory(env::WaveEnv)
    traj = CircularArraySARTTrajectory(
        capacity = num_steps(env),
        state = Vector{TotalWaveSol} => (),
        action = Vector{typeof(env.total_dynamics.design(0.0f0))} => ())

    return traj
end

dim = TwoDim(5.0f0, 256)
dynamics_kwargs = Dict(:dim => dim, :pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

cyl = Cylinder(0.0f0, 0.0f0, 0.5f0, 0.1f0)

tmax = 5.0f0

env = WaveEnv(
    initial_condition = Pulse(dim, -4.0f0, 0.0f0, 10.0f0),
    wave = build_wave(dim, fields = 6),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = cyl,
    space = design_space(cyl, 1.0f0),
    design_steps = 100,
    tmax = tmax;
    dynamics_kwargs...) |> gpu

action = rand(action_space(env))

traj = episode_trajectory(env)
policy = RandomPolicy(action_space(env))
agent = Agent(policy, traj)

data = SaveData()
@time run(agent, env, StopWhenDone(), data)

# sol = TotalWaveSol(traj.traces.state...)
# @time render!(sol.total, path = "total.mp4")
# @time render!(env.sol.incident, path = "incident.mp4")
# @time render!(sol.scattered, path = "scattered.mp4")

sol = TotalWaveSol(data.sols...)
actions = DesignTrajectory(data.designs...)

@time render!(sol.total, actions, path = "total.mp4")
