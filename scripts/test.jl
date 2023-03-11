using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)

using Waves

dim = TwoDim(5.0f0, 512)
dynamics_kwargs = Dict(:dim => dim, :pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
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

data = SaveData()
@time run(policy, env, StopWhenDone(), data)

sol = TotalWaveSol(data.sols...)
actions = DesignTrajectory(data.designs...)

@time render!(sol.total, actions, path = "total.mp4")
