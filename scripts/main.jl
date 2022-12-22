using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using GLMakie
using Waves

wave = Wave2D(x_min = -5.0, x_max = 5.0, y_min = -5.0, y_max = 5.0, t_max = 10.0)
M = 2
@parameters x_pos[1:M], y_pos[1:M]
@parameters wavespeed

function C(x, y, t)

    s = wavespeed

    for i in 1:M
        s = IfElse.ifelse((x - x_pos[i]) ^ 2 + (y - y_pos[i]) ^ 2 < 1.0 ^ 2, 0.0, s)
    end

    return s
end

sim = WaveSimulation(wave,
    ic = (x, y) -> exp(- 1 * ((x - 2.5) ^ 2 + (y - 0.0) ^ 2)),
    C = C,
    n = 30,
    p = [
        wavespeed => 4.0,
        Pair.(collect(x_pos), [-3.0, -3.0])...,
        Pair.(collect(y_pos), [-2.0, 2.0])...]
    )

sol = ∫ₜ(sim, dt = 0.01)

fig = Figure(resolution = (1920, 1080), fontsize = 20)

ax = Axis3(
    fig[1,1],
    aspect = (1, 1, 1),
    perspectiveness = 0.5,
    title="3D Wave",
    xlabel = "X",
    ylabel = "Y",
    zlabel = "Z",
    )

xlims!(ax, getbounds(wave.x)...)
ylims!(ax, getbounds(wave.y)...)
zlims!(ax, 0.0, 5.0)

record(fig, "func.mp4", axes(sol.u, 3)) do i
    GLMakie.empty!(ax.scene)
    GLMakie.surface!(ax, sol.x, sol.y, sol.u[:, :, i], colormap = :ice, shading = false)
end