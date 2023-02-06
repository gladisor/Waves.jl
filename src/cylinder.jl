export Cylinder, speed

struct Cylinder <: Scatterer
    x
    y
    r
    c
end

function GLMakie.mesh!(ax::GLMakie.Axis3, cyl::Cylinder)
    GLMakie.mesh!(ax, GLMakie.GeometryBasics.Cylinder3{Float32}(GLMakie.Point3f(cyl.x, cyl.y, -1.0), GLMakie.Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

function Base.:∈(xy::Tuple, cyl::Cylinder)
    return ((xy[1] - cyl.x) ^ 2 + (xy[2] - cyl.y) ^ 2) <= cyl.r ^ 2
end

function Base.:+(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x + cyl2.x, cyl1.y + cyl2.y, cyl1.r + cyl2.r, cyl1.c + cyl2.c)
end

function Base.:-(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.x - cyl2.x, cyl1.y - cyl2.y, cyl1.r - cyl2.r, cyl1.c - cyl2.c)
end

function Base.:*(cyl::Cylinder, m::Float64)
    return Cylinder(cyl.x * m, cyl.y * m, cyl.r * m, cyl.c * m)
end

function Base.:*(m::Float64, cyl::Cylinder)
    return cyl * m
end

function speed(dim::TwoDim, cyl::Cylinder)
    C = zeros(size(dim))

    for i ∈ axes(C, 1)
        for j ∈ axes(C, 2)
            if (dim.x[i], dim.y[j]) ∈ cyl
                C[i, j] = 0.8
            end
        end
    end

    return C
end