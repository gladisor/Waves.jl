using CairoMakie, BSON, Flux
using Plots

pml_error = BSON.load("rebuttal/pml_error.bson")
no_pml_error = BSON.load("no_pml_error.bson")
node_error = BSON.load("rebuttal/node_error.bson")

# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "Prediction Horizon (Actions)", ylabel = "Log of Prediction Error")
# scatter!(ax, pml_error[:horizon], log.(Flux.mean.(pml_error[:error])), label = "Ours (PML)")
# scatter!(ax, no_pml_error[:horizon], log.(Flux.mean.(no_pml_error[:error])), label = "Ours (No PML)")
# scatter!(ax, node_error[:horizon], log.(Flux.mean.(node_error[:error])), label = "NeuralODE")
# axislegend(ax, position = :lt)
# save("error.png", fig)

horizon = pml_error[:horizon]
pml = pml_error[:error]
no_pml = no_pml_error[:error]
node = node_error[:error]

scale_func = x -> x
# scale_func = x -> log(x)
pml_mean = scale_func.(Flux.mean.(pml))
pml_std = scale_func.(Flux.std.(pml))
pml_low = pml_mean .- pml_std
pml_high = pml_mean .+ pml_std

no_pml_mean = scale_func.(Flux.mean.(no_pml))
no_pml_std = scale_func.(Flux.std.(no_pml))
no_pml_low = no_pml_mean .- no_pml_std
no_pml_high = no_pml_mean .+ no_pml_std

node_mean = scale_func.(Flux.mean.(node))
node_std = scale_func.(Flux.std.(node))
node_low = node_mean .- node_std
node_high = node_mean .+ node_std

alpha = 0.2
fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = "Prediction Horizon (Actions)", 
    ylabel = "Long-Term Prediction Error",
    title = "Effect of Increased Prediction Horizon on Error")
# CairoMakie.ylims!(ax, -1.0, 300.0)

# no_pml_color = :green
# lines!(ax, horizon, no_pml_mean, color = no_pml_color, label = "Ours (No PML)")
# band!(ax, horizon, no_pml_low, no_pml_high, color = (no_pml_color, alpha))

node_color = :red
lines!(ax, horizon, node_mean, color = node_color, label = "NeuralODE", linewidth = 3)
band!(ax, horizon, node_low, node_high, color = (node_color, alpha))

pml_color = :green
lines!(ax, horizon, pml_mean, color = pml_color, label = "Ours (PML)", linewidth = 3)
band!(ax, horizon, pml_low, pml_high, color = (pml_color, alpha))

axislegend(ax, position = :lt)
save("error.png", fig)