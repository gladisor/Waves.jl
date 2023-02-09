export build_pml

function build_pml(dim::OneDim, width::Float64)
    x = abs.(dim.x)

    start = min(x[1], x[end]) - width

    pml = zeros(size(dim))

    for i ∈ axes(pml, 1)
        pml[i] = max(x[i] - start, 0.0) / width
    end

    clamp!(pml, 0.0, 1.0)
    return pml .^ 2
end

"""
Assuming an x axis which is symmetric build a vector which contains zeros in the
interior and slowly scales from zero to one at the edges.
"""
function build_pml(dim::TwoDim, width::Float64)
    x, y = abs.(dim.x), abs.(dim.y)

    start_x = min(x[1], x[end]) - width
    start_y = min(y[1], y[end]) - width

    pml = zeros(size(dim))

    for i ∈ axes(pml, 1)
        for j ∈ axes(pml, 2)
            depth = maximum([x[i] - start_x, y[j] - start_y, 0.0]) / width
            pml[i, j] = depth
        end
    end

    clamp!(pml, 0.0, 1.0)
    return pml .^ 2
end

function build_pml(dim::ThreeDim, width::Float64)
    x, y, z = abs.(dim.x), abs.(dim.y), abs.(dim.z)

    start_x = x[end] - width
    start_y = y[end] - width
    start_z = z[end] - width

    pml = zeros(size(dim))

    for i ∈ axes(pml, 1)
        for j ∈ axes(pml, 2)
            for k ∈ axes(pml, 3)
                depth = maximum([x[i] - start_x, y[j] - start_y, z[k] - start_z, 0.0]) / width
                pml[i, j, k] = depth
            end
        end
    end

    return pml .^ 2
end
