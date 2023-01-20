export Cylinder

"""
Structure which holds the information about a cylindrical scatterer.
"""
struct Cylinder <: AbstractDesign
    x
    y
    r
    c
end

"""
Constructor for Cylinder which only specifies x and y position.

```
cyl = Cylinder(0.0, 0.0)
```
"""
function Cylinder(x, y)
    return Cylinder(x, y, 1.0, 0.0)
end

function Base.range(start::Cylinder, stop::Cylinder, length::Int) #::Vector{Cylinder}
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
    return Cylinder(ps...)
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
    return Cylinder(cyl1.x + cyl2.x, cyl1.y + cyl2.y, cyl1.r + cyl2.r, cyl1.c + cyl2.c)
end

function perturb(cyl::Cylinder, action::Cylinder, dim::TwoDim)
    x_min, x_max = getbounds(dim.x)
    y_min, y_max = getbounds(dim.y)
    new_cyl = cyl + action
    x = clamp(new_cyl.x + new_cyl.r, x_min, x_max)
    y = clamp(new_cyl.y + new_cyl.r, y_min, y_max)
    return Cylinder(x, y, new_cyl.r, new_cyl.c)
end

function reset!(cyl::Cylinder, dim::TwoDim)
    x_min, x_max = getbounds(dim.x)
    y_min, y_max = getbounds(dim.y)
    x = rand(Uniform(x_min + cyl.r, x_max - cyl.r))
    y = rand(Uniform(y_min + cyl.r, y_max - cyl.r))
    return Cylinder(x, y, cyl.r, cyl.c)
end

function GLMakie.mesh!(ax::GLMakie.Axis3, cyl::Cylinder)
    GLMakie.mesh!(
        ax, 
        GLMakie.GeometryBasics.Cylinder3{Float32}(GLMakie.Point3f(cyl.x, cyl.y, -1.0), GLMakie.Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

function interpolate(initial::Cylinder, final::Cylinder, t::Num)
    ps = interpolate.(design_parameters(initial), design_parameters(final), t)
    return Cylinder(ps...)
end

function Base.:∈(xy::Tuple, cyl::Cylinder)
    x, y = xy
    return ((x - cyl.x)^2 + (y - cyl.y)^2) < cyl.r ^ 2
end

function wave_speed(wave::Wave{TwoDim}, design::ParameterizedDesign{Cylinder})::Function

    C = (x, y, t) -> begin
        cyl = interpolate(design.initial, design.final, get_t_norm(design, t))
        return IfElse.ifelse((x, y) ∈ cyl, cyl.c, wave.speed)
    end
    
    return C
end