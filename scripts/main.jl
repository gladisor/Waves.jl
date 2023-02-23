using Flux
using Waves
using CairoMakie
using ReinforcementLearning

dx = 0.1f0
ambient_speed = 1.0f0
dt = Waves.stable_dt(dx, ambient_speed)
dim = TwoDim(15.0f0, dx)
config = Scatterers(M = 4, r = 1.0f0, disk_r = 8.0f0, c = 0.0f0)

kwargs = Dict(:dim => dim, :pml_width => 4.0f0, :pml_scale => 10.0f0, :ambient_speed => ambient_speed, :dt => dt)

env = WaveEnv(
    initial_condition = Pulse([-9.0f0, 9.0f0], 1.0f0),
    dyn = WaveDynamics(design = config; kwargs...), 
    design_space = Waves.design_space(config, 1.0f0),
    design_steps = 5, 
    tmax = 10.0f0)

policy = RandomPolicy(action_space(env))

run(policy, env, StopWhenDone())