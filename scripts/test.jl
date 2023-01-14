using Waves

## simulation hyperparameters
gs = 5.0
kwargs = Dict(:tmax => 20.0, :speed => 1.0, :n => 30, :dt => 0.05)
dp = (0.0, 0.0, 0.7, 0.2)

## important objects
wave = Wave(dim = TwoDim(gs))
pulse = GaussianPulse(intensity = 1.0, loc = [2.5, 2.5])
design = Cylinder(dp...)

## build env
env = WaveEnv(wave = wave, ic = pulse, design = design, design_steps = 40; kwargs...)
reset!(env)
env.design.design = Cylinder(dp...)
steps = Vector{typeof(env.design.design)}([env.design.design])

## run env with random control
while !is_terminated(env)
    action = Cylinder(randn(), randn(), 0.0, 0.0)
    [push!(steps, s) for s ∈ Waves.step(env, action)]
end

## render total field
sol_p_total = WaveSol(env.sim)
steps = vcat(steps...)
@time render!(sol_p_total, design = steps, path = "p_total.mp4")

## generate incident field and render scattered field
p_inc = WaveSim(wave = wave, ic = pulse; kwargs...)
Waves.step!(p_inc)
sol_p_inc = WaveSol(p_inc)
sol_p_sc = sol_p_total - sol_p_inc
@time render!(sol_p_sc, design = steps, path = "p_sc.mp4")