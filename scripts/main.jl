using Waves
using ReinforcementLearning
using Flux

dim = TwoDim(8.0f0, 0.025f0)
cyl = Cylinder([0.0f0, 0.0f0], 1.0f0, 0.2f0)
kwargs = Dict(:dim => dim, :pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 1.0f0, :dt => 0.02f0)
dyn = WaveDynamics(design = cyl; kwargs...)
wave = Wave(dim, 6)

env = gpu(WaveEnv(
    initial_condition = Pulse(dim, 3.0f0, 3.0f0, 10.0f0),
    iter = WaveIntegrator(wave, split_wave_pml, runge_kutta, dyn),
    design_space = design_space(cyl, 0.5f0),
    design_steps = 50,
    tmax = 20.0f0))

policy = RandomDesignPolicy(action_space(env))

data = SaveData()
@time run(policy, env, StopWhenDone(), data)

sol_tot = vcat(data.sols...)
designs = vcat(data.designs...)

wave = env.initial_condition(wave)
dyn_inc = WaveDynamics(;kwargs...)
iter_inc = gpu(WaveIntegrator(wave, split_wave_pml, runge_kutta, dyn_inc))
@time sol_inc = cpu(integrate(iter_inc, length(sol_tot) - 1))
sol_sc = sol_tot - sol_inc
@time render!(sol_sc, designs, path = "vid_sc.mp4")