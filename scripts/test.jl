using Waves

function Base.:-(sol::WaveSol, other::WaveSol)
    data = sol.data .- other.data
    return WaveSol(sol.wave, sol.dims, data)
end

grid_size = 5.0
wave = Wave(dim = TwoDim(-grid_size, grid_size, -grid_size, grid_size))
pulse = GaussianPulse(intensity = 1.0, loc = [2.5, 2.5])
design = ParameterizedDesign(Cylinder(-3.0, -3.0, 0.5, 0.0))
p_total = WaveSim(wave = wave, ic = pulse, design = design, t_max = 20.0, speed = 1.0, n = 30, dt = 0.05)
env = WaveEnv(p_total, design, 40)
reset!(env)
env.design = ParameterizedDesign(Cylinder(0.0, 0.0, 1.0, 0.2))
steps = Vector{typeof(env.design.design)}([env.design.design])

@time while !is_terminated(env)
    action = Cylinder(randn(), randn(), 0.0, 0.0)
    [push!(steps, s) for s ∈ Waves.step(env, action)]
end

sol_p_total = WaveSol(env.sim)
steps = vcat(steps...)
@time render!(sol_p_total, design = steps, path = "p_total.mp4")

p_inc = WaveSim(wave = wave, ic = pulse, t_max = 20.0, speed = 1.0, n = 30, dt = 0.05)
Waves.step!(p_inc)
sol_p_inc = WaveSol(p_inc)
sol_p_sc = sol_p_total - sol_p_inc
@time render!(sol_p_sc, path = "p_sc.mp4")