using GLMakie

using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using IfElse
using LinearAlgebra

using Waves

struct Cylinder
    x::Float32
    y::Float32
    r::Float32
    c::Float32
end

function design_parameters(cyl::Cylinder)
    return [cyl.x, cyl.y, cyl.r, cyl.c]
end

function GLMakie.mesh!(ax::Axis3, cyl::Cylinder)
    GLMakie.mesh!(ax, GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(cyl.x, cyl.y, 0.), Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

cyl1 = Cylinder(-5.0f0, 6.0f0, 1.0f0, 0.0f0)
cyl2 = Cylinder(-5.0f0, 3.0f0, 1.0f0, 0.0f0)
cyl3 = Cylinder(-5.0f0, 0.0f0, 1.0f0, 0.0f0)
cyl4 = Cylinder(-5.0f0, -3.0f0, 1.0f0, 0.0f0)
cyl5 = Cylinder(-5.0f0, -6.0f0, 1.0f0, 0.0f0)
design = [cyl1, cyl2, cyl3, cyl4, cyl5]

@parameters cyl1_x, cyl1_y, cyl1_r, cyl1_c
@parameters cyl2_x, cyl2_y, cyl2_r, cyl2_c
@parameters cyl3_x, cyl3_y, cyl3_r, cyl3_c
@parameters cyl4_x, cyl4_y, cyl4_r, cyl4_c
@parameters cyl5_x, cyl5_y, cyl5_r, cyl5_c

@parameters wave_c

cyl1_params = [cyl1_x, cyl1_y, cyl1_r, cyl1_c]
cyl2_params = [cyl2_x, cyl2_y, cyl2_r, cyl2_c]
cyl3_params = [cyl3_x, cyl3_y, cyl3_r, cyl3_c]
cyl4_params = [cyl4_x, cyl4_y, cyl4_r, cyl4_c]
cyl5_params = [cyl5_x, cyl5_y, cyl5_r, cyl5_c]

wave = Wave2D(x_min = -10.0, x_max = 10.0, y_min = -10.0, y_max = 10.0, t_max = 10.0)

wavespeed(x, y, t) = IfElse.ifelse(
    (x - cyl1_x - t) ^ 2 + (y - cyl1_y) ^ 2 < cyl1_r ^ 2, cyl1_c,
    IfElse.ifelse(
        (x - cyl2_x - t) ^ 2 + (y - cyl2_y) ^ 2 < cyl2_r ^ 2, cyl2_c,
        IfElse.ifelse(
            (x - cyl3_x - t) ^ 2 + (y - cyl3_y) ^ 2 < cyl3_r ^ 2, cyl3_c,
            IfElse.ifelse(
                (x - cyl4_x - t) ^ 2 + (y - cyl4_y) ^ 2 < cyl4_r ^ 2, cyl4_c,
                IfElse.ifelse(
                    (x - cyl5_x - t) ^ 2 + (y - cyl5_y) ^ 2 < cyl5_r ^ 2, cyl5_c,
                    wave_c
                )
            )
        )
    )
)

sim = WaveSimulation(
    wave,
    ic = (x, y) -> exp(- 5 * ((x - 2.5)^2 + (y - 0.0)^2)),
    C = wavespeed,
    n = 30,
    p = [
        wave_c => 2.0,
        Pair.(cyl1_params, design_parameters(cyl1))...,
        Pair.(cyl2_params, design_parameters(cyl2))...,
        Pair.(cyl3_params, design_parameters(cyl3))...,
        Pair.(cyl4_params, design_parameters(cyl4))...,
        Pair.(cyl5_params, design_parameters(cyl5))...
        ]);

sim.prob = remake(
    sim.prob, 
    tspan = (0.0, 10.0), 
    p = [
        2.0, ## ambient wavespeed
        design_parameters(cyl1)...,
        design_parameters(cyl2)...,
        design_parameters(cyl3)...,
        design_parameters(cyl4)...,
        design_parameters(cyl5)...,
    ])

@time sol = ∫ₜ(sim, dt = 0.05)

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

record(fig, "vid_makie.mp4", axes(sol.u, 3)) do i
    GLMakie.empty!(ax.scene)
    design = [cyl1, cyl2, cyl3, cyl4, cyl5]
    design = [Cylinder(cyl.x + sol.t[i], cyl.y, cyl.r, cyl.c) for cyl ∈ design]

    for cyl ∈ design
        mesh!(ax, cyl)
    end

    GLMakie.surface!(ax, sol.x, sol.y, sol.u[:, :, i], colormap = :ice, shading = false)
end