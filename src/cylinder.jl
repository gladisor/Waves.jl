export Cylinder, speed

struct Cylinder <: Scatterer
    x
    y
    r
    c
end

function Cylinder(dim::TwoDim; r, c, offset = 0.0)
    x = rand(Uniform(dim.x[1] + r + offset, dim.x[end] - r - offset))
    y = rand(Uniform(dim.y[1] + r + offset, dim.y[end] - r - offset))
    return Cylinder(x, y, r, c)
end

function Base.:+(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x + cyl2.x, cyl1.y + cyl2.y, cyl1.r + cyl2.r, cyl1.c + cyl2.c)
end

function Base.:-(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x - cyl2.x, cyl1.y - cyl2.y, cyl1.r - cyl2.r, cyl1.c - cyl2.c)
end

function Base.:*(cyl::Cylinder, n::Float64)
    return Cylinder(cyl.x * n, cyl.y * n, cyl.r * n, cyl.c * n)
end

function Base.:*(n::Float64, cyl::Cylinder)
    return cyl * n
end

function Base.:/(cyl::Cylinder, n::Float64)
    return Cylinder(cyl.x / n, cyl.y / n, cyl.r / n, cyl.c / n)
end

function Base.:∈(xy::Tuple, cyl::Cylinder)
    return ((xy[1] - cyl.x) ^ 2 + (xy[2] - cyl.y) ^ 2) <= cyl.r ^ 2
end

function speed(cyl::Cylinder, g::AbstractArray{<: AbstractFloat, 3}, ambient_speed)
    pos = cu([cyl.x ;;; cyl.y])
    in_cyl = dropdims(sum((g .- pos) .^ 2, dims = 3) .< cyl.r ^ 2, dims = 3)
    return .~ in_cyl .* ambient_speed .+ in_cyl * cyl.c
end