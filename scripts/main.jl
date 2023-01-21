using ModelingToolkit
using Waves

# function Waves.open_boundary(wave::Wave{OneDim})::Vector{Equation}
#     x, t, u = Waves.unpack(wave)
#     x_min, x_max = getbounds(x)
#     bcs = vcat(Waves.absorbing_condition(wave), [Waves.time_condition(wave)])
#     bcs[1] = u(x_min, t) ~ sin(t) / 2
#     return bcs
# end

# function Waves.open_boundary(wave::Wave{TwoDim})::Vector{Equation}
#     x, y, t, u = Waves.unpack(wave)
#     x_min, x_max = getbounds(x)
#     bcs = vcat(Waves.absorbing_condition(wave), [Waves.time_condition(wave)])
#     bcs[1] = u(x_min, y, t) ~ sin(3*t) / 5
#     return bcs
# end

function Base.length(sol::WaveSol)
    return length(sol.data)
end

kwargs = Dict(:tmax => 40.0, :ambient_speed => 1.0, :n => 21, :dt => 0.05)
gs = 5.0
# dim = OneDim(-gs, gs)
dim = TwoDim(gs)
design = ParameterizedDesign(Cylinder(0.0, 0.0, 0.5, 0.0))
wave = Wave(dim = dim)

wave
ic = Waves.Silence()
sim = WaveSim(
    wave = wave, 
    design = design, 
    ic = ic; kwargs...)
# env = WaveEnv(sim, design, 20)
# reset!(env)
# reset!(env.design.design, env.sim.wave.dim)
# steps = Vector{typeof(env.design.design)}([env.design.design])

# ## run env with random control
# while !is_terminated(env)
#     action = Cylinder(randn()*0.5, randn()*0.5, 0.0, 0.2)
#     s = perturb(env, action)

#     [push!(steps, s[i]) for i ∈ axes(s, 1)]
# end

# sol_tot = WaveSol(env)

# render!(sol_tot, design = steps, path = "silence_cyl_2d.mp4")

# sim_inc = WaveSim(wave = wave, ic = ic, open = true; kwargs...)

# Waves.step!(sim_inc)
# sol_inc = WaveSol(sim_inc)
# sol_sc = sol_tot - sol_inc
# render!(sol_sc, design = steps, path = "silence_cyl_2d_sc.mp4")
# plot_energy!(sol_inc, sol_sc, path = "silence_cyl_2d_energy.png")