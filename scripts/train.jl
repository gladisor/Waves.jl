using GLMakie

using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using Flux

using Waves

wave = Wave1D(x_min = -10.0, x_max = 10.0, t_max = 10.0)

@parameters pulse_1_x, pulse_2_x
ic = (x) -> exp(- 5 * (x - pulse_1_x) ^ 2) - exp(-5 * (x - pulse_2_x) ^ 2)

sim = WaveSimulation(
    wave,
    ic = ic,
    C = (x, t) -> 4.0,
    n = 100,
    p = [
        pulse_1_x => 1.0,
        pulse_2_x => -1.0,
    ])

sol = ∫ₜ(sim, dt = 0.01)

fig = Figure(resolution = (1920, 1080), fontsize = 20)
ax = Axis(fig[1,1], title = "Wave", xlabel = "X")

xlims!(ax, getbounds(wave.x)...)
ylims!(ax, -1.0, 1.0)

record(fig, "vid.mp4", axes(sol.u, 2)) do i
    GLMakie.empty!(ax.scene)
    lines!(ax, sol.x, sol.u[:, i], color = :blue)
end
