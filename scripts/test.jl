using Waves
using Flux
using CairoMakie

dim = OneDim(5.0f0, 200)
pulse = Pulse(dim, 0.0f0, 5.0f0)
wave = pulse(zeros(Float32, size(dim)..., 2))

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 15.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)
dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)

n = 100

opt = Descent(0.01)
ps = Flux.params(wave)

fig = Figure(resolution = (1920, 1080), fontsize = 40)
ax = Axis(fig[1, 1], xlabel = "Spacial domain (m)", ylabel = "Displacement", title = "Before Optimization")
xlims!(ax, dim.x[1], dim.x[end])
ylims!(ax, -1.0f0, 1.0f0)
lines!(ax, dim.x, displacement(wave), color = :blue, linewidth = 5, label = "Initial Pulse")
final_wave = integrate(cell, wave, dynamics, n)[end]
lines!(ax, dim.x, displacement(final_wave), color = :orange, linewidth = 5, label = "Pulse After $n Timesteps")
axislegend(ax, position = :lb)

for i ∈ 1:100
    Waves.reset!(dynamics)

    gs = Flux.gradient(ps) do 

        u = integrate(cell, wave, dynamics, n)
        e = sum(energy(displacement(u[end])))

        Flux.ignore() do 
            println(e)
        end

        return e
    end

    Flux.Optimise.update!(opt, ps, gs)
end

Waves.reset!(dynamics)
ax2 = Axis(fig[1, 2], xlabel = "Spacial domain (m)", ylabel = "Displacement", title = "After Optimization")
xlims!(ax2, dim.x[1], dim.x[end])
ylims!(ax2, -1.0f0, 1.0f0)
lines!(ax2, dim.x, displacement(wave), color = :blue, linewidth = 5, label = "Initial Pulse")
final_wave = integrate(cell, wave, dynamics, n)[end]
lines!(ax2, dim.x, displacement(final_wave), color = :orange, linewidth = 5, label = "Pulse After $n Timesteps")
axislegend(ax2, position = :lb)

save("u.png", fig)