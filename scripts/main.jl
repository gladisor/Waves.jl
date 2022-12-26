using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using GLMakie

using Waves

# abstract type AbstractDesign end

# struct Cylinder <: AbstractDesign
#     x
#     y
#     r
#     c
# end

# function Cylinder(x, y)
#     return Cylinder(x, y, 1.0, 0.0)
# end

# function design_parameters(design::Cylinder)
#     return [design.x, design.y, design.r, design.c]
# end

# struct Configuration
#     scatterers::Vector{Cylinder}
# end

# function design_parameters(design::Configuration)
#     return design_parameters.(design.scatterers)
# end

# struct Boat <: AbstractDesign
#     x
#     y
#     θ
# end

# function design_parameters(design::Boat)
#     return [design.x, design.y, design.θ]
# end

wave = Wave(dim = TwoDim(-10., 10., -10., 10.))
@named sys = wave(
    ic = (x, y) -> exp(-1.0 * (x ^ 2 + y ^ 2)),
    speed = 2.0)

n = 30
disc = MOLFiniteDifference([Pair.(Waves.dims(wave), n)...], wave.t)
prob = discretize(sys, disc)
grid = get_discrete(sys, disc)

"""
For some reason solve! is not stopping at the tstop but going all the way through
"""

iter = init(
    prob, 
    Tsit5(), 
    advance_to_tstop = true, 
    # stop_at_next_tstop = true
    saveat = 0.05
    )

add_tstop!(iter, 5.0)

iter.p[1] = 2.0
step!(iter)
iter.p[1] = 0.5
step!(iter)

for i ∈ iter
    println(iter.t)
end

sol = iter.sol[grid[wave.u(Waves.dims(wave)..., wave.t)]]

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

xlims!(ax, getbounds(wave.dim.x)...)
ylims!(ax, getbounds(wave.dim.y)...)
zlims!(ax, 0.0, 3.0)

record(fig, "3d.mp4", axes(sol, 1)) do i
    empty!(ax.scene)
    surface!(
        ax, 
        collect(grid[wave.dim.x]),
        collect(grid[wave.dim.y]),
        sol[i], 
        colormap = :ice, 
        shading = false)
end