using Waves
using Flux
using CairoMakie

ep = Episode(path = "episode.bson")
length(ep)

horizon = 1
data = Flux.DataLoader(prepare_data([ep], horizon), batchsize = -1, shuffle = true, partial = false)
s, a, t, y = first(data)

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0f0)
heatmap!(ax1, s.dim.x, s.dim.y, s.wave[:, :, 1], colormap = :ice)
ax2 = Axis(fig[1, 2], aspect = 1.0f0)
heatmap!(ax2, s.dim.x, s.dim.y, s.wave[:, :, 2], colormap = :ice)
ax3 = Axis(fig[1, 3], aspect = 1.0f0)
heatmap!(ax3, s.dim.x, s.dim.y, s.wave[:, :, 3], colormap = :ice)
save("wave.png", fig)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, t, y[:, 1], label = "Total")
lines!(ax, t, y[:, 2], label = "Incident")
lines!(ax, t, y[:, 3], label = "Scattered")
axislegend(ax)
save("signal.png", fig)