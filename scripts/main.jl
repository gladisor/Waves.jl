using ModelingToolkit
using Waves

gs = 5.0
# dim = OneDim(-gs, gs)
dim = TwoDim(gs)
cyl = ParameterizedDesign(Cylinder(0.0, 0.0))
wave_inc = Wave(dim = dim)
# ic = GaussianPulse(intensity = 1.0)
ic = Silence()

function Waves.open_boundary(wave::Wave{OneDim})::Vector{Equation}
    x, t, u = Waves.unpack(wave)
    x_min, x_max = getbounds(x)
    bcs = vcat(Waves.absorbing_condition(wave), [Waves.time_condition(wave)])
    bcs[1] = u(x_min, t) ~ sin(t) / 2
    return bcs
end

function Waves.open_boundary(wave::Wave{TwoDim})::Vector{Equation}
    x, y, t, u = Waves.unpack(wave)
    x_min, x_max = getbounds(x)
    bcs = vcat(Waves.absorbing_condition(wave), [Waves.time_condition(wave)])
    bcs[1] = u(x_min, y, t) ~ sin(3*t) / 5
    return bcs
end

sim_inc = WaveSim(
    wave = wave_inc, 
    ic = ic,
    open = true, 
    tmax = 20.0,
    ambient_speed = 1.0,
    n = 21,
    dt = 0.05)

Waves.step!(sim_inc)
sol_inc = WaveSol(sim_inc)
render!(sol_inc, path = "silence_2d.mp4")

## simulation hyperparameters
# gs = 5.0
# kwargs = Dict(:tmax => 40.0, :speed => 1.0, :n => 21, :dt => 0.05)
# dp = (0.0, 0.0, 0.7, 0.0)
# pulse = GaussianPulse(intensity = 1.0, loc = [2.5, 2.5])

# ## important objects
# wave_inc = Wave(dim = TwoDim(gs), free = true)
# wave_tot = Wave(dim = TwoDim(gs), free = true)

# sim_inc = WaveSim(wave = wave_inc, ic = pulse; kwargs...)
# Waves.step!(sim_inc)
# sol_inc = WaveSol(sim_inc)

# design = Cylinder(dp...)
# ## build env
# env = WaveEnv(wave = wave_tot, ic = pulse, design = design, design_steps = 20; kwargs...)

# reset!(env)
# random_position!(env.design.design, env.sim.wave.dim)
# steps = Vector{typeof(env.design.design)}([env.design.design])

# ## run env with random control
# while !is_terminated(env)
#     action = Cylinder(randn(), randn(), 0.0, 0.0)
#     [push!(steps, s) for s ∈ Waves.step(env, action)]
# end

# ## render total field
# sol_tot = WaveSol(env.sim)
# steps = vcat(steps...)
# sol_sc = sol_tot - sol_inc

# render!(sol_tot, design = steps, path = "wave_tot.mp4")
# render!(sol_sc, design = steps, path = "wave_sc.mp4")
# # render!(sol_inc, path = "wave_inc.mp4")
# plot_energy!(sol_inc, sol_sc, path = "wave.png")