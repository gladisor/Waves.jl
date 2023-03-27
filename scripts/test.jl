using CairoMakie

x1 = range(0, 2pi, 100)
y1 = sin.(x1)

x2 = range(2pi, 4pi, 100)
y2 = cos.(x2)

fig = Figure()
ax = Axis(fig[1, 1])

lines!(ax, vcat(x1, x2), vcat(y1, y2), label = "1")
axislegend(ax)

save("u.png", fig)