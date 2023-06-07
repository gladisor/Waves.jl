using DataFrames
using CSV
using CairoMakie

mpc = CSV.read("pml=on,latent=15m,horizon=3/data.csv", DataFrame)
random = CSV.read("random_baseline/data.csv", DataFrame)

tspan = mpc[!, :tspan]
mpc_avg = (mpc[!, :sigma1] .+ mpc[!, :sigma2] .+ mpc[!, :sigma3]) ./ 3.0f0
random_avg = (random[!, :sigma1] .+ random[!, :sigma2] .+ random[!, :sigma3]) ./ 3.0f0

fig = Figure()
ax = Axis(
    fig[1, 1], 
    title = "Comparison of Scattered Energy Produced Over Time",
    xlabel = "Time (s)",
    ylabel = "Scattered Energy (σ)")
    
lines!(ax, tspan, mpc_avg, color = :blue, label = "mpc")
lines!(ax, tspan, random_avg, color = :orange, label = "random")
axislegend(ax, position = :rb)
save("sigma.png", fig)

