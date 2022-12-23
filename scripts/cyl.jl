
struct Cylinder
    x::Real
    y::Real
    r::Real
    c::Real
end

function Cylinder(x, y)
    return Cylinder(x, y, 1.0, 0.0)
end

struct PositionAction
    dx::Real
    dy::Real
end

function Base.+(cyl::Cylinder, a::PositionAction)
    return Cylinder(cyl.x + a.dx, cyl.y + a.dy, cyl.r, cyl.c)
end

struct RadiusAction
    dr::Real
end

function Base.+(cyl::Cylinder, a::RadiusAction)
    return Cylinder(cyl.x, cyl.y, cyl.r + a.dr, cyl.c)
end

struct WaveSpeedAction
    dc::Real
end

function Base.+(cyl::Cylinder, a::WaveSpeedAction)
    return Cylinder()
end