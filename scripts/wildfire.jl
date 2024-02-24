using Waves, Flux, CairoMakie

Flux.device!(0)
Flux.CUDA.allowscalar(false)

dim = TwoDim(100.0f0, 1024)
f = build_normal(
    build_grid(dim), 
    [
        0.0f0   0.0f0;
        10.0f0 -10.0f0
    ], 
    [1.0f0, 1.0f0], 
    [1.0f0, 1.0f0])

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0)
heatmap!(ax, dim.x, dim.y, f)
save("wildfire.png", fig)