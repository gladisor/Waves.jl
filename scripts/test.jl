using Waves

gs = 5.0
wave = Wave(dim = TwoDim(-gs, gs, -gs, gs))
pulse = GaussianPulse(intensity = 1.0, loc = [2.5, 2.5])
kwargs = Dict(:t_max => 20.0, :speed => 1.0, :n => 30, :dt => 0.05)
cyl_params = [0.0, 0.0, 0.7, 0.2]

env = WaveEnv(
    WaveSim(wave = wave, ic = pulse, design = design; kwargs...),
    ParameterizedDesign(Cylinder(cyl_params...)), 
    40)

reset!(env)
env.design = ParameterizedDesign(Cylinder(cyl_params...))
steps = Vector{typeof(env.design.design)}([env.design.design])

@time while !is_terminated(env)
    action = Cylinder(randn(), randn(), 0.0, 0.0)
    [push!(steps, s) for s ∈ Waves.step(env, action)]
end

sol_p_total = WaveSol(env.sim)
steps = vcat(steps...)
@time render!(sol_p_total, design = steps, path = "p_total.mp4")

p_inc = WaveSim(wave = wave, ic = pulse; kwargs...)
Waves.step!(p_inc)
sol_p_inc = WaveSol(p_inc)
sol_p_sc = sol_p_total - sol_p_inc
@time render!(sol_p_sc, design = steps, path = "p_sc.mp4")