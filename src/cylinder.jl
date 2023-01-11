export Cylinder

"""
Structure which holds the information about a cylindrical scatterer with a constant
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

function Base.range(start::Cylinder, stop::Cylinder, length::Int)
    x = collect.(range.(design_parameters(start), design_parameters(stop), length))
    return [Cylinder(ps...) for ps in collect(zip(x...))]
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

function Waves.wave_speed(wave::Wave{TwoDim}, design::ParameterizedDesign{Cylinder})::Function

    C = (x, y, t) -> begin
        t_norm = (t - design.t_initial) / (design.t_final - design.t_initial)
        x_interp = design.initial.x + (design.final.x - design.initial.x) * t_norm
        y_interp = design.initial.y + (design.final.y - design.initial.y) * t_norm

        return IfElse.ifelse(
            (x - x_interp) ^ 2 + (y - y_interp) ^ 2 < (design.final.r - design.initial.r) * t_norm, 
            (design.final.c - design.initial.c) * t_norm, 
            wave.speed)
    end
    
    return C
end