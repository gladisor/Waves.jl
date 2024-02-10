using CairoMakie, BSON, Flux
using Loess

pml_error = BSON.load("results/variable_source_results/our_error.bson")
node_error = BSON.load("results/variable_source_results/node_error.bson")
pinn_error = BSON.load("results/variable_source_results/pinn_error.bson")

horizon = pml_error[:horizon]
pml = pml_error[:error]
node = node_error[:error]
pinn = pinn_error[:error]

pml_mean = Flux.mean.(pml)
pml_interp = loess(horizon, pml_mean)
pml_mean_smooth = predict(pml_interp, horizon)
pml_std = Flux.std.(pml)
pml_low = pml_mean_smooth .- 1.92 * pml_std / sqrt(32)
pml_high = pml_mean_smooth .+ 1.92 * pml_std / sqrt(32)

node_mean = Flux.mean.(node)
node_interp = loess(horizon, node_mean)
node_mean_smooth = predict(node_interp, horizon)
node_std = Flux.std.(node)
node_low = node_mean_smooth .- 1.92 * node_std / sqrt(32)
node_high = node_mean_smooth .+ 1.92 * node_std / sqrt(32)

pinn_mean = Flux.mean.(pinn)
pinn_interp = loess(horizon, pinn_mean)
pinn_mean_smooth = predict(pinn_interp, horizon)
pinn_std = Flux.std.(pinn)
pinn_low = pinn_mean_smooth .- 1.92 * pinn_std / sqrt(32)
pinn_high = pinn_mean_smooth .+ 1.92 * pinn_std / sqrt(32)

alpha = 0.1
# fig = Figure(resolution = (1600, 1200), fontsize = 50)
fig =  Figure()
ax = Axis(fig[1, 1], 
    xlabel = "Prediction Horizon (Actions)", 
    ylabel = "Long-Term Prediction Error",
    title = "Effect of Increased Prediction Horizon on Error")

node_color = :red
lines!(ax, horizon, node_mean_smooth, color = node_color, label = "NODE", linewidth = 3)
band!(ax, horizon, node_low, node_high, color = (node_color, alpha))
pml_color = :green
lines!(ax, horizon, pml_mean_smooth, color = pml_color, label = "cPILS - Numerical Integration", linewidth = 3)
band!(ax, horizon, pml_low, pml_high, color = (pml_color, alpha))
pinn_color = :purple
lines!(ax, horizon, pinn_mean_smooth, color = pinn_color, label = "cPILS - PINN", linewidth = 3)
band!(ax, horizon, pinn_low, pinn_high, color = (pinn_color, alpha))
axislegend(ax, position = :lt)
save("error.png", fig)