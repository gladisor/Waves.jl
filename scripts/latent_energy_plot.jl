pml_z = cpu(generate_latent_solution(pml_model, s, a, t))
no_pml_z = cpu(generate_latent_solution(no_pml_model, s, a, t))

pml_incident_energy = vec(sum(pml_z[:, 1, :, 1] .^ 2, dims = 1))
no_pml_incident_energy = vec(sum(no_pml_z[:, 1, :, 1] .^ 2, dims = 1))

pml_total_energy = vec(sum(pml_z[:, 2, :, 1] .^ 2, dims = 1))
no_pml_total_energy = vec(sum(no_pml_z[:, 2, :, 1] .^ 2, dims = 1))

pml_scattered_energy = vec(sum((pml_z[:, 2, :, 1] .- pml_z[:, 1, :, 1]) .^ 2, dims = 1))
no_pml_scattered_energy = vec(sum((no_pml_z[:, 2, :, 1] .- no_pml_z[:, 1, :, 1]) .^ 2, dims = 1))

fig = Figure()
ax1 = Axis(fig[1, 1], ylabel = "Incident Energy")
lines!(ax1, tspan, pml_incident_energy, label = "PML")
lines!(ax1, tspan, no_pml_incident_energy, label = "No PML")

ax2 = Axis(fig[2, 1], ylabel = "Total Energy")
lines!(ax2, tspan, pml_total_energy, label = "PML")
lines!(ax2, tspan, no_pml_total_energy, label = "No PML")

ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "Scattered Energy")
lines!(ax3, tspan, pml_scattered_energy, label = "PML")
lines!(ax3, tspan, no_pml_scattered_energy, label = "No PML")

axislegend(ax1, position = :lt)
# axislegend(ax2, position = :lt)
# axislegend(ax3, position = :lt)
save("energy.png", fig)