using Waves
using CairoMakie

M = 4

r = fill(1.0f0, M)
c = fill(AIR * 3, M)

low_pos = fill(-15.0f0, M, 2)
high_pos = fill(15.0f0, M, 2)

low = AdjustablePositionScatterers(Cylinders(low_pos, r, c))
high = AdjustablePositionScatterers(Cylinders(high_pos, r, c))

design_space = DesignSpace(low, high)

a = rand(design_space)

dim = TwoDim(15.0f0, 700)
grid = build_grid(dim)

speed_sound = speed(a, grid, WATER)






fig  = Figure()
ax = Axis(fig[1, 1], aspect = 1.0)
# mesh!(ax, a)
CairoMakie.heatmap!(ax, dim.x, dim.y, speed_sound)
save("cylinders.png", fig)


