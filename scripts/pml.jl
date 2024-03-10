using Waves, Flux, CairoMakie

Flux.device!(0)
dim = TwoDim(5.0f0, 512)
grid = build_grid(dim)
dyn = AcousticDynamics(dim, WATER, 1.0f0, 0.0f0)
iter = gpu(Integrator(runge_kutta, dyn, 1f-5))
tspan = gpu(build_tspan(iter, 0.0f0, 600))

wave = build_wave(dim, 12)
ic = build_normal(grid, [0.0f0 0.0f0], [0.3f0], [1.0f0])
wave[:, :, 1] .= ic
wave[:, :, 7] .= ic
wave = gpu(wave)

C = t -> WATER
F = NoSource()
sol = cpu(iter(wave, tspan, [C, F]))

tspan = cpu(tspan)

colorrange = (-1.0f0, 1.0f0)

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0, title = "T = $(tspan[1])s")
heatmap!(ax, dim.x, dim.y, sol[:, :, 1, 1], colormap = :ic, colorrange = colorrange)
ax = Axis(fig[1, 2], aspect = 1.0, title = "T = $(tspan[end ÷ 4])s")
heatmap!(ax, dim.x, dim.y, sol[:, :, 1, end ÷ 4], colormap = :ice, colorrange = colorrange)
ax = Axis(fig[1, 3], aspect = 1.0, title = "T = $(tspan[3 * end ÷ 4])s")
heatmap!(ax, dim.x, dim.y, sol[:, :, 1, 3 * end ÷ 4], colormap = :ice, colorrange = colorrange)
ax = Axis(fig[1, 4], aspect = 1.0, title = "T = $(tspan[end])s")
heatmap!(ax, dim.x, dim.y, sol[:, :, 1, end], colormap = :ice, colorrange = colorrange)

# ax = Axis(fig[2, 1], aspect = 1.0, title = "T = $(tspan[1])s")
# heatmap!(ax, dim.x, dim.y, sol_pml[:, :, 1, 1], colormap = :ice, colorrange = colorrange)
# ax = Axis(fig[2, 2], aspect = 1.0, title = "T = $(tspan[end ÷ 4])s")
# heatmap!(ax, dim.x, dim.y, sol_pml[:, :, 1, end ÷ 4], colormap = :ice, colorrange = colorrange)
# ax = Axis(fig[2, 3], aspect = 1.0, title = "T = $(tspan[3 * end ÷ 4])s")
# heatmap!(ax, dim.x, dim.y, sol_pml[:, :, 1, 3 * end ÷ 4], colormap = :ice, colorrange = colorrange)
# ax = Axis(fig[2, 4], aspect = 1.0, title = "T = $(tspan[end])s")
# heatmap!(ax, dim.x, dim.y, sol_pml[:, :, 1, end], colormap = :ice, colorrange = colorrange)
save("wave.png", fig)