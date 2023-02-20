export Cylinder, action_space

struct Cylinder <: Scatterer
    pos::AbstractVector{Float32}
    r::Float32
    c::Float32
end

function Cylinder(dim::TwoDim; r::Float32, c::Float32, offset::Float32 = 0.0f0)
    x = rand(Uniform(dim.x[1] + r + offset, dim.x[end] - r - offset))
    y = rand(Uniform(dim.y[1] + r + offset, dim.y[end] - r - offset))
    return Cylinder([x, y], r, c)
end

function Base.:+(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.pos .+ cyl2.pos, cyl1.r + cyl2.r, cyl1.c + cyl2.c)
end

function Base.:-(cyl1::Cylinder, cyl2::Cylinder)
    return Cylinder(cyl1.pos .- cyl2.pos, cyl1.r - cyl2.r, cyl1.c - cyl2.c)
end

function Base.:*(cyl::Cylinder, n::AbstractFloat)
    return Cylinder(cyl.pos * n, cyl.r * n, cyl.c * n)
end

function Base.:*(n::AbstractFloat, cyl::Cylinder)
    return cyl * n
end

function Base.:/(cyl::Cylinder, n::Float32)
    return Cylinder(cyl.pos / n, cyl.r / n, cyl.c / n)
end

function Base.zero(::Cylinder)
    return Cylinder(zeros(Float32, 2), 0.0f0, 0.0f0)
end

"""
Generates a bit matrix which is the size of the domain containing ones
where the cylinder is present and zeros elsewhere.
"""
function location_mask(cyl::Cylinder, g::AbstractArray{Float32, 3})
    pos = reshape(cyl.pos, 1, 1, 2)
    return dropdims(sum((g .- pos) .^ 2, dims = 3) .< cyl.r ^ 2, dims = 3)
end

function speed(cyl::Cylinder, g::AbstractArray{Float32, 3}, ambient_speed)
    loc = location_mask(cyl, g)
    return .~ loc .* ambient_speed .+ loc * cyl.c
end

function Base.rand(cyl::ClosedInterval{Cylinder})

    pos = rand.(Uniform.(cyl.left.pos, cyl.right.pos))

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

    return Cylinder(pos, r, c)
end

function action_space(::Cylinder, scale::Float32)
    return Cylinder([-scale, -scale], 0.0f0, 0.0f0)..Cylinder([scale, scale], 0.0f0, 0.0f0)
end

function Flux.gpu(cyl::Cylinder)
    return Cylinder(gpu(cyl.pos), cyl.r, cyl.c)
end

function Flux.cpu(cyl::Cylinder)
    return Cylinder(cpu(cyl.pos), cyl.r, cyl.c)
end