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

function speed(dim::TwoDim, cyl::Cylinder, C0::Matrix{Float64})
    C = ones(size(dim)) .* C0

    for i ∈ axes(C, 1)
        for j ∈ axes(C, 2)
            if (dim.x[i], dim.y[j]) ∈ cyl
                C[i, j] = cyl.c
            end
        end
    end

    return C
end