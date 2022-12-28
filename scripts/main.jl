using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse
using ModelingToolkit.Symbolics: CallWithMetadata
using GLMakie

using Waves

abstract type AbstractDesign end

"""
Structure which holds the information about a cylindrical scattere with a constant
internal wavespeed: c
"""
struct Cylinder <: AbstractDesign
    x
    y
    r
    c
    name
end

"""
Constructor for Cylinder which only specifies x and y position.

```
cyl = Cylinder(0.0, 0.0)
```
"""
function Cylinder(x, y; name = "")
    return Cylinder(x, y, 1.0, 0.0, name)
end


function Cylinder(x, y, r, c)
    return Cylinder(x, y, r, c, "")
end

"""
Constructor for creating a purely parameterized Cylinder design. All attributes are
parameters instead of taking on a value.

```
@named cyl = Cylinder()
```
"""
function Cylinder(;name)
    x = Symbol(name, "_x")
    y = Symbol(name, "_y")
    r = Symbol(name, "_r")
    c = Symbol(name, "_c")

    ps = @parameters $x, $y, $r, $c
    return Cylinder(ps..., name)
end

"""
Retrieves pointers to each parameter of the cylinder design.
"""
function design_parameters(design::Cylinder)
    return [design.x, design.y, design.r, design.c]
end

"""
Defines addition between two Cylindrical scatterers. Simply adds each parameter and
assigns an empty string as name.
"""
function Base.:+(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x + cyl2.x, cyl1.y + cyl2.y, cyl1.r + cyl2.r, cyl1.c + cyl2.c, "")
end

function GLMakie.mesh!(ax::Axis3, cyl::Cylinder)
    GLMakie.mesh!(ax, GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(cyl.x, cyl.y, 0.), Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

mutable struct ParameterizedDesign{D <: AbstractDesign}
    design::D
    initial::D
    final::D
    t_initial::Num
    t_final::Num
end

function ParameterizedDesign(;design, initial, final)
    @parameters t_initial, t_final
    return ParameterizedDesign(design, initial, final, t_initial, t_final)
end

function wave_speed()
end

function Waves.wave_equation(wave::Wave{TwoDim}, pd::ParameterizedDesign{Cylinder})

    C = (x, y, t) -> begin
        t_norm = (t - pd.t_initial) / (pd.t_final - pd.t_initial)
        x_interp = pd.initial.x + (pd.final.x - pd.initial.x) * t_norm
        y_interp = pd.initial.y + (pd.final.y - pd.initial.y) * t_norm

        return IfElse.ifelse((x - x_interp) ^ 2 + (y - y_interp) ^ 2 < 1.0, 0.0, wave.speed)
    end

    return Waves.wave_equation(wave, C)
end

design = Cylinder(-5.0, 0.0)
@named initial = Cylinder()
@named final = Cylinder()

pd = ParameterizedDesign(
    design = design,
    initial = initial,
    final = final)

action = Cylinder(10.0, 0.0, 0.0, 0.0)
new_design = pd.design + action
wave = Wave(dim = TwoDim(-10., 10., -10., 10.))

ps = [
    wave.speed => 2.0,
    (design_parameters(pd.initial) .=> design_parameters(pd.design))...,
    (design_parameters(pd.final) .=> design_parameters(new_design))...,
    pd.t_initial => 0.0,
    pd.t_final => 1.0]

eq = Waves.wave_equation(wave, pd)

bcs = [
    wave.u(dims(wave)..., 0.0) ~ exp(-1.0 * (wave.dim.x ^ 2 + wave.dim.y ^ 2)),
    Waves.boundary_conditions(wave)...
    ]

@named sys = PDESystem(
    eq, bcs, Waves.get_domain(wave, t_max = 5.0), [dims(wave)..., wave.t], 
    [wave.u(dims(wave)..., wave.t)], ps)

n = 30
disc = MOLFiniteDifference([Pair.(Waves.dims(wave), n)...], wave.t)
prob = discretize(sys, disc)
grid = get_discrete(sys, disc)
iter = init(prob, Tsit5(), advance_to_tstop = true, saveat = 0.05)

# # for i ∈ 3:5
# #     add_tstop!(iter, iter.t + dt)
# #     step!(iter)
# #     iter.p[2:3] = iter.p[4:5]
# #     iter.p[4:5] .= [cyls[i].x, cyls[i].y]
# #     iter.p[end-1] = iter.p[end]
# #     iter.p[end] = iter.t + dt
# # end

# # # step!(iter)

# # sol = iter.sol[grid[wave.u(Waves.dims(wave)..., wave.t)]]

# # fig = Figure(resolution = (1920, 1080), fontsize = 20)
# # ax = Axis3(
# #     fig[1,1],
# #     aspect = (1, 1, 1),
# #     perspectiveness = 0.5,
# #     title="3D Wave",
# #     xlabel = "X",
# #     ylabel = "Y",
# #     zlabel = "Z",
# #     )

# # xlims!(ax, getbounds(wave.dim.x)...)
# # ylims!(ax, getbounds(wave.dim.y)...)
# # zlims!(ax, 0.0, 3.0)

# # record(fig, "3d.mp4", axes(sol, 1)) do i
# #     empty!(ax.scene)
# #     surface!(
# #         ax, 
# #         collect(grid[wave.dim.x]),
# #         collect(grid[wave.dim.y]),
# #         sol[i], 
# #         colormap = :ice, 
# #         shading = false)
# # end