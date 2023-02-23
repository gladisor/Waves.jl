using Flux
using Waves
using CairoMakie

function stable_dt(dx::Float32, ambient_speed::Float32)::Float32
    return sqrt(dx^2 / ambient_speed^2)
end

dx = 0.1f0
ambient_speed = 1.0f0
dt = stable_dt(dx, ambient_speed)

dim = TwoDim(15.0f0, dx)
cyl = Cylinder([0.0f0, 0.0f0], 1.0f0, 0.0f0)
config = Scatterers(M = 4, r = 1.0f0, disk_r = 8.0f0, c = 0.0f0)

dyn = WaveDynamics(
    dim = dim, 
    design = config,
    pml_width = 4.0f0,
    pml_scale = 10.0f0,
    ambient_speed = ambient_speed,
    dt = dt)

ic = Pulse([-9.0f0, 9.0f0], 1.0f0)

g = dyn.g
pos = reshape(ic.pos, 1, 1, size(ic.pos)...)
u = exp.(- dropdims(sum((g .- pos) .^ 2, dims = 3), dims = 3))
u = cat(u, zeros(Float32, size(u)..., 2), dims = 3)

n = 300
sol = integrate(u, dyn, n)
println(sol.t)
traj = DesignTrajectory(dyn, n)
render!(sol, traj, path = "vid.mp4")

# p = WavePlot(dim)
# heatmap!(p.ax, dim.x, dim.y, u, colormap = :ice)
# save("speed.png", p.fig)