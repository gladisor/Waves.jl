export Cylinder, pos_action_space

struct Cylinder <: Scatterer
    x::Float32
    y::Float32
    r::Float32
    c::Float32
end

# function Cylinder(dim::TwoDim; r, c, offset = 0.0)
#     x = rand(Uniform(dim.x[1] + r + offset, dim.x[end] - r - offset))
#     y = rand(Uniform(dim.y[1] + r + offset, dim.y[end] - r - offset))
#     return Cylinder(x, y, r, c)
# end

function Base.:+(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x + cyl2.x, cyl1.y + cyl2.y, cyl1.r + cyl2.r, cyl1.c + cyl2.c)
end

function Base.:-(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x - cyl2.x, cyl1.y - cyl2.y, cyl1.r - cyl2.r, cyl1.c - cyl2.c)
end

function Base.:*(cyl::Cylinder, n::Float32)
    return Cylinder(cyl.x * n, cyl.y * n, cyl.r * n, cyl.c * n)
end

function Base.:*(n::Float32, cyl::Cylinder)
    return cyl * n
end

function Base.:/(cyl::Cylinder, n::Float32)
    return Cylinder(cyl.x / n, cyl.y / n, cyl.r / n, cyl.c / n)
end

function Base.:∈(xy::Tuple, cyl::Cylinder)
    return ((xy[1] - cyl.x) ^ 2 + (xy[2] - cyl.y) ^ 2) <= cyl.r ^ 2
end

function Base.zero(::Cylinder)
    return Cylinder(0.0, 0.0, 0.0, 0.0)
end

function speed(cyl::Cylinder, g::AbstractArray{Float32, 3}, ambient_speed)
    pos = gpu([cyl.x ;;; cyl.y])
    in_cyl = dropdims(sum((g .- pos) .^ 2, dims = 3) .< cyl.r ^ 2, dims = 3)
    return .~ in_cyl .* ambient_speed .+ in_cyl * cyl.c
end

function Base.rand(cyl::ClosedInterval{Cylinder})
    x = rand(Uniform(cyl.left.x, cyl.right.x))
    y = rand(Uniform(cyl.left.y, cyl.right.y))

    if cyl.right.r > cyl.left.r
        r = rand(Uniform(cyl.left.r, cyl.right.r))
    else
        r = 0.0
    end

    if cyl.right.c > cyl.left.c
        c = rand(Uniform(cyl.left.c, cyl.right.c))
    else
        c = 0.0
    end

    return Cylinder(x, y, r, c)
end

function pos_action_space(::Cylinder, scale::Float32)
    return Cylinder(-scale, -scale, 0.0, 0.0)..Cylinder(scale, scale, 0.0, 0.0)
end