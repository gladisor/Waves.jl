using Flux
using Waves
using CairoMakie

dx = 0.05f0
ambient_speed = 1.0f0
dt = sqrt(dx^2 / ambient_speed^2)
dim = TwoDim(15.0f0, dx)
g = grid(dim)
grad = Waves.gradient(dim.x)
cyl = Cylinder([0.0f0, 0.0f0], 1.0f0, 0.1f0)
design = DesignInterpolator(cyl)
pml = build_pml(dim, 4.0f0, 10.0f0)
dyn = WaveDynamics(dim, g, grad, design, pml, ambient_speed, 0, dt)

p = WavePlot(dim)
heatmap!(p.ax, dim.x, dim.y, Waves.speed(dyn, 0.0f0), colormap = :ice)
save("speed.png", p.fig)

# dyn = WaveDynamics(dim, grad, C, pml, 0, dt)
# g = grid(dim)

# u = exp.(-(dropdims(sum(g .* [1.0f0 ;;; 0.0f0], dims = 3), dims = 3) .+ 9.0f0) .^ 2)
# u = cat(u, zeros(Float32, (size(u)..., 2)), dims = 3)

# # initial_condition = Pulse([-9.0f0, 9.0f0], 1.0f0)
# # u = initial_condition(dim)

# sol = integrate(u, dyn, 200)
# render!(sol, path = "test.mp4")

# gs = 10.0f0
# dx = 0.05f0
# ambient_speed = 1.0f0
# dt = sqrt(dx^2/ambient_speed^2)
# tmax = 20.0f0
# n = Int(round(tmax / dt))
# M = 10
# pml_width = 4.0f0

# config = Scatterers(M = M, r = 0.5f0, disk_r = gs - pml_width, c = 0.1f0)
# kwargs = Dict(:dim => TwoDim(gs, dx), :pml_width => pml_width, :pml_scale => 20.0f0, :ambient_speed => ambient_speed, :dt => dt)
# dyn = WaveDynamics(design = config; kwargs...)

# env = gpu(WaveEnv(
#     initial_condition = Pulse([-9.0f0, 9.0f0], 1.0f0), 
#     dyn = dyn, design_steps = 20, tmax = tmax))

# policy = RandomPolicy(action_space(env, 1.0f0))
# @time run(policy, env, StopWhenDone())

# dim = kwargs[:dim]
# p = WavePlot(dim)
# heatmap!(p.ax, dim.x, dim.y, env.sol.u[end][:, :, 1], colormap = :ice)
# save("env.png", p.fig)