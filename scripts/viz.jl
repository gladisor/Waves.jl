tick_length = length(energy(sol_tot))
old_ticks = collect(1:100:tick_length)
new_ticks = collect(range(0, Waves.get_tmax(env.sim), length = length(old_ticks)))

fig = Figure(resolution = (1920, 1080), fontsize = 50)
ax = Axis(fig[1, 1], 
    title = "Scattered Wave Energy Over Time",
    xlabel = "Time dt = 0.05",
    ylabel = "Wave Energy: Σx²",
    xticks = (old_ticks,  string.(new_ticks)))


# lines!(ax, energy(sol_tot), linewidth = 8, label = "Total", linestyle = :solid)
lines!(ax, energy(sol_inc), linewidth = 8, label = "Incident")
lines!(ax, energy(sol_sc), linewidth = 8, label = "Scattered")

Legend(fig[1, 2], ax, "Wave")
save("wave.png", fig)