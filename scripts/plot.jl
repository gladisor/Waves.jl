using CairoMakie
using CairoMakie: Point
using Waves

dim = TwoDim(10.0, 400)
u = pulse(dim)

cyl = Circle(Point(2.0, 2.0), 0.5)
fig = Figure(resolution = (1920, 1080))
ax = Axis(fig[1, 1], aspect = AxisAspect(1.0))
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, dim.y[1], dim.y[end])

heatmap!(ax, dim.x, dim.y, u, colormap = :ice)
poly!(ax, cyl, color = :gray)
save("heat.png", fig)