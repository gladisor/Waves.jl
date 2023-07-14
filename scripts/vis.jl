# using DataFrames
# using CSV
using CairoMakie
using BSON

# mpc = CSV.read("pml=on,latent=15m,horizon=3/data.csv", DataFrame)
# random = CSV.read("random_baseline/data.csv", DataFrame)

# tspan = mpc[!, :tspan]
# mpc_avg = (mpc[!, :sigma1] .+ mpc[!, :sigma2] .+ mpc[!, :sigma3]) ./ 3.0f0
# random_avg = (random[!, :sigma1] .+ random[!, :sigma2] .+ random[!, :sigma3]) ./ 3.0f0

# fig = Figure()
# ax = Axis(
#     fig[1, 1], 
#     title = "Comparison of Scattered Energy Produced Over Time",
#     xlabel = "Time (s)",
#     ylabel = "Scattered Energy (σ)")
    
# lines!(ax, tspan, mpc_avg, color = :blue, label = "mpc")
# lines!(ax, tspan, random_avg, color = :orange, label = "random")
# axislegend(ax, position = :rb)
# save("sigma.png", fig)

data = BSON.load("loss.bson")
horizon = data[:horizon]
pml_loss = data[:pml_loss]
no_pml_loss = data[:no_pml_loss]

xs = vec(ones(Int, size(pml_loss)) .* horizon')

dodge = ones(Int, size(pml_loss))
dodge1 = vec(dodge)
dodge2 = 2 * vec(dodge)

fig = Figure()
ax = Axis(fig[1, 1], 
    title = "Effect of Increased Planning Horizon on Validation Loss",
    xlabel = "Planning Horizon", 
    ylabel = "Validation Loss")

boxplot!(
    ax, 
    xs, 
    vec(pml_loss),
    dodge = dodge1, 
    show_outliers = false,
    color = :red,
    width = 3,
    label = "PML")

boxplot!(
    ax,
    xs, 
    vec(no_pml_loss),
    dodge = dodge2,
    show_outliers = false,
    color = :blue,
    width = 3,
    label = "No PML")

axislegend(ax, position = :lt)
save("loss.png", fig)

