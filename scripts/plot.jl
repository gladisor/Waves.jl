using CairoMakie, BSON

y = BSON.load("/home/012761749/Waves.jl/results/variable_source_results/ground_truth.bson")[:y]
node_y_hat = BSON.load("/home/012761749/Waves.jl/results/variable_source_results/node.bson")[:y_hat]
our_y_hat = BSON.load("/home/012761749/Waves.jl/results/variable_source_results/ours.bson")[:y_hat]
pinn_y_hat = BSON.load("/home/012761749/Waves.jl/results/variable_source_results/pinn.bson")[:y_hat]
t = build_tspan(0.0f0, 1f-5, 20000)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Scattered Energy", title = "Variable Source Location Scattered Energy Prediction With Random Control")
lines!(ax, t, y, label = "Ground Truth")
lines!(ax, t, our_y_hat, color = (:green, 0.6), label = "Ours (PML)")
lines!(ax, t, node_y_hat, color = (:red, 0.6), label = "NeuralODE")
lines!(ax, t, pinn_y_hat, color = (:purple, 0.6), label = "PINC")
axislegend(ax, position = :lt)
save("prediction.png", fig)