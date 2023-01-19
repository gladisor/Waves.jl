using Waves

## simulation hyperparameters
gs = 5.0
kwargs = Dict(:tmax => 40.0, :speed => 1.0, :n => 21, :dt => 0.05)
dp = (0.0, 0.0, 0.7, 0.0)

pulse = GaussianPulse(intensity = 1.0, loc = [2.5, 2.5])

## important objects
wave_inc = Wave(dim = TwoDim(gs), free = true)
wave_tot = Wave(dim = TwoDim(gs), free = true)

sim_inc = WaveSim(wave = wave_inc, ic = pulse; kwargs...)
Waves.step!(sim_inc)
sol_inc = WaveSol(sim_inc)

design = Cylinder(dp...)
## build env
env = WaveEnv(wave = wave_tot, ic = pulse, design = design, design_steps = 20; kwargs...)

reset!(env)
random_position!(env.design.design, env.sim.wave.dim)
steps = Vector{typeof(env.design.design)}([env.design.design])

## run env with random control
while !is_terminated(env)
    action = Cylinder(randn(), randn(), 0.0, 0.0)
    [push!(steps, s) for s ∈ Waves.step(env, action)]
end

## render total field
sol_tot = WaveSol(env.sim)
steps = vcat(steps...)
sol_sc = sol_tot - sol_inc

render!(sol_tot, design = steps, path = "wave_tot.mp4")
render!(sol_sc, design = steps, path = "wave_sc.mp4")
# render!(sol_inc, path = "wave_inc.mp4")
plot_energy!(sol_inc, sol_sc, path = "wave.png")