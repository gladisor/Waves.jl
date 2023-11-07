using CairoMakie, BSON, Flux

pml_error = BSON.load("rebuttal/pml_error.bson")
no_pml_error = BSON.load("no_pml_error.bson")
node_error = BSON.load("rebuttal/node_error.bson")

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Prediction Horizon", ylabel = "Log of Prediction Error")
scatter!(ax, pml_error[:horizon], log.(Flux.mean.(pml_error[:error])), label = "Ours (PML)")
scatter!(ax, no_pml_error[:horizon], log.(Flux.mean.(no_pml_error[:error])), label = "Ours (No PML)")
scatter!(ax, node_error[:horizon], log.(Flux.mean.(node_error[:error])), label = "NODE")
axislegend(ax, position = :lt)
save("error.png", fig)

# using Plots
# pyplot()
# xs = range(1,20,step=1)
# μs1 = log.(xs)
# σs1 = rand(length(xs))
# μs2 = log10.(xs)
# σs2 = rand(length(xs))

# p = Plots.plot(xs,μs1,grid=false,ribbon=σs1, shape=:circle, label="μs1")
# Plots.plot!(p, xs,μs2,grid=false,ribbon=σs2, shape=:square, label="μs2")

# savefig(p,"file.png")
