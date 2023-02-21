using Flux
using Waves
using ReinforcementLearning

# mutable struct <: AbstractHook
# end

gs = 15.0f0
dx = 0.1f0
ambient_speed = 1.0f0
dt = sqrt(dx^2/ambient_speed^2)
tmax = 20.0f0
n = Int(round(tmax / dt))
M = 10
pml_width = 4.0f0

config = Scatterers(M = M, r = 0.5f0, disk_r = gs - pml_width, c = 0.1f0)
kwargs = Dict(:dim => TwoDim(gs, dx), :pml_width => pml_width, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)
dyn = WaveDynamics(design = config; kwargs...)

env = WaveEnv(
    initial_condition = Pulse([-9.0f0, 9.0f0], 1.0f0), 
    dyn = dyn, 
    design_steps = 5, 
    tmax = tmax) |> gpu

policy = RandomPolicy(action_space(env, 1.0f0))
@time run(policy, env, StopWhenDone())