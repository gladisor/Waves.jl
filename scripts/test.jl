using Waves

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

design = ParameterizedDesign(Cylinder(0.0, 0.0, 0.5, 0.0))
kwargs = Dict(
    :wave => wave, :ic => Silence(), :boundary => PlaneWave(), 
    :ambient_speed => 2.0, :tmax => 10.0, :n => 21, :dt => 0.05)

@time sim_tot = WaveSim(
    design = design
    ;kwargs...)
@time sim_inc = WaveSim(;kwargs...)

@time Waves.propagate!(sim_tot)
@time Waves.propagate!(sim_inc)

@time sol_tot = WaveSol(sim_tot)
@time sol_inc = WaveSol(sim_inc)

steps = range(design.design, design.design, length(sol_tot))
sol_sc = sol_tot - sol_inc

@time render!(sol_tot, 
    design = steps, 
    path = "sol_tot.mp4")

@time render!(sol_sc, 
    design = steps, 
    path = "sol_sc.mp4")

@time Waves.plot_energy!(sol_inc = sol_inc, sol_sc = sol_sc, path = "inc.png")