using CairoMakie
using Waves

dim = TwoDim(10.0, 400)
u = pulse(dim)

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, dim.x, dim.y, u)
save("heat.png", fig)