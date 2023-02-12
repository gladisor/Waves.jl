using GLMakie
using DifferentialEquations

using Waves
using Waves: ∇x, ∇y

# gs = 10.0
# Δ = 0.01
# dim = TwoDim(gs, Δ)
# u = gaussian_pulse(dim, 0.0, 0.0, 1.0)[:, :, 1]
# @time dux = ∇x(u, Δ)
# @time duy = ∇y(u, Δ)
# @time duxx = ∇x(dux, Δ)
# @time duyy = ∇y(duy, Δ)
# duxxyy = duxx .+ duyy

# fig = Figure(resolution = (1920, 1080))
# ax = Axis3(fig[1, 1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Displacement (m)")
# zlims!(ax, -1.0, 1.0)
# surface!(ax, dim.x, dim.y, duxxyy, colormap = :ice)
# GLMakie.save("func.png", fig)


gs = 10.0
Δ = 0.1
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)

dim = TwoDim(gs, Δ)
u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
C = WaveSpeed(dim, C0)
pml = build_pml(dim, pml_width) * pml_scale
prob = ODEProblem(split_wave!, u0, tspan, [Δ, C, pml])
@time sol = solve(prob, RK4())
;