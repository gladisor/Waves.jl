using Waves
using ReinforcementLearning
using Flux

dim = TwoDim(8.0f0, 0.05f0)
cyl = Cylinder([3.0f0, 3.0f0], 1.0f0, 0.2f0)
kwargs = Dict(:dim => dim, :pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 1.0f0, :dt => 0.02f0)
dyn = WaveDynamics(design = cyl; kwargs...)
wave = Wave(dim, 6)

env = gpu(WaveEnv(
    initial_condition = Pulse(dim, 0.0f0, 0.0f0, 10.0f0),
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
@time render!(sol_inc, designs, path = "vid_inc.mp4")
@time render!(sol_sc, designs, path = "vid_sc.mp4")

laplace = laplacian(dim.x)
mask = circle_mask(dim, 7.0f0)

inc_flux = flux(sol_inc, laplace, mask)
sc_flux = flux(sol_sc, laplace, mask)

println("Plotting Flux")
fig = Figure()
ax = Axis(fig[1, 1])
xlims!(ax, sol_inc.t[1], sol_inc.t[end])
lines!(ax, sol_inc.t, inc_flux, color = :blue, label = "Incident")
lines!(ax, sol_sc.t, sc_flux, color = :red, label = "Scattered")
axislegend(ax)
save("flux.png", fig)
