using Waves
using CairoMakie

dim = TwoDim(5.0f0, 200)
pulse1 = Pulse(dim, 0.0f0, 2.0f0, 5.0f0)
wave = zeros(Float32, size(dim)..., 6)
wave = pulse1(wave)

n = 600

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 50.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)
cyl = Cylinder([0.0f0, -2.0f0], 1.0f0, 0.1f0)
dynamics = WaveDynamics(dim = dim, design = cyl; dynamics_kwargs...)
action = Cylinder([0.0f0, 4.0f0], 0.0f0, 0.0f0)
dynamics.design = DesignInterpolator(cyl, action, 0.0f0, n * dynamics.dt)

traj = DesignTrajectory(dynamics, n)

@time u = integrate(cell, wave, dynamics, n)
pushfirst!(u, wave)
t = collect(range(0.0f0, dynamics.dt * n, n + 1))
sol = WaveSol(dim, t, u)
render!(sol, traj, path = "vid.mp4")

fig = Figure(
    resolution = (1920, 1080), 
    fontsize = 40)

ax1 = Axis(fig[1, 1], title = "T = $(t[1])", aspect = AxisAspect(1.0))
xlims!(ax1, dim.x[1], dim.x[end])
ylims!(ax1, dim.y[1], dim.y[end])
heatmap!(ax1, dim.x, dim.y, displacement(u[1]), colormap = :ice)
mesh!(ax1, dynamics.design(0.0f0))

ax2 = Axis(fig[1, 2], title = "T = $(t[end ÷ 2])", aspect = AxisAspect(1.0))
xlims!(ax2, dim.x[1], dim.x[end])
ylims!(ax2, dim.y[1], dim.y[end])
heatmap!(ax2, dim.x, dim.y, displacement(u[end ÷ 2]), colormap = :ice)
mesh!(ax2, dynamics.design((n * dynamics.dt) / 2))

ax3 = Axis(fig[1, 3], title = "T = $(t[end])", aspect = AxisAspect(1.0))
xlims!(ax3, dim.x[1], dim.x[end])
ylims!(ax3, dim.y[1], dim.y[end])
heatmap!(ax3, dim.x, dim.y, displacement(u[end]), colormap = :ice)
mesh!(ax3, dynamics.design(n * dynamics.dt))

save("sim.png", fig)