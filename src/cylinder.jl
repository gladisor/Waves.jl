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

function Cylinder(dim::TwoDim; r = 1.0, c = 0.0)
    x_min, x_max = getbounds(dim.x)
    y_min, y_max = getbounds(dim.y)
    x = rand(Uniform(x_min + r, x_max - r))
    y = rand(Uniform(y_min + r, y_max - r))
    return Cylinder(x, y, r, c)
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
Required function for the interface which allows us to interpolate between two cylinders with an
arbatrary number of points. These interpolations are used for plotting
"""
function Base.range(start::Cylinder, stop::Cylinder, length::Int)
    x = collect.(range.(design_parameters(start), design_parameters(stop), length))
    return [Cylinder(ps...) for ps in collect(zip(x...))]
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

    x = clamp(new_cyl.x, x_min + new_cyl.r, x_max - new_cyl.r)
    y = clamp(new_cyl.y, y_min + new_cyl.r, y_max - new_cyl.r)
    return Cylinder(x, y, new_cyl.r, new_cyl.c)
end

function GLMakie.mesh!(ax::GLMakie.Axis3, cyl::Cylinder)
    GLMakie.mesh!(ax, GLMakie.GeometryBasics.Cylinder3{Float32}(GLMakie.Point3f(cyl.x, cyl.y, -1.0), GLMakie.Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

"""
Checks if a point in two dimensions is withing the scatterer
"""
function Base.:∈(xy::Tuple, cyl::Cylinder)
    x, y = xy
    return ((x - cyl.x)^2 + (y - cyl.y)^2) < cyl.r ^ 2
end

function Base.:/(cyl::Cylinder, n::Real)
    return Cylinder(cyl.x/n, cyl.y/n, cyl.r, cyl.c)
end

function (C::WaveSpeed{TwoDim, Design{Cylinder}})(x, y, t)
    design = C.design

    t′ = (t - design.ti) / (design.tf - design.ti)
    x′, y′, r′, c′ = Waves.interpolate.(design_parameters(design.initial), design_parameters(design.final), t′)
    inside = (x - x′) ^ 2 + (y - y′) ^ 2 < r′^2
    return inside * c′ + (1 - inside) * wave.speed
end