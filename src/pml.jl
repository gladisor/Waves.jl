export build_pml

function build_pml(dim::OneDim, width::Float32, scale::Float32)
    x = abs.(dim.x)
    start = min(x[1], x[end]) - width
    pml = zeros(Float32, size(dim))
    for i ∈ axes(pml, 1)
        pml[i] = max(x[i] - start, 0.0f0) / width
    end
    clamp!(pml, 0.0f0, 1.0f0)
    return pml .^ 2 * scale
end

"""
Assuming an x axis which is symmetric build a vector which contains zeros in the
interior and slowly scales from zero to one at the edges.
"""
function build_pml(dim::TwoDim, width::Float32, scale::Float32)
    x = abs.(dim.x)
    pml_start = x[1] - width
    pml_region = x .> pml_start
    x[.~ pml_region] .= 0.0f0
    x[pml_region] .= (x[pml_region] .- minimum(x[pml_region])) / width
    x = repeat(x, 1, length(dim.y))
    return x .^ 2 * scale
end

function build_pml(dim::ThreeDim, width::Float32, scale::Float32)
    x, y, z = abs.(dim.x), abs.(dim.y), abs.(dim.z)

    start_x = x[end] - width
    start_y = y[end] - width
    start_z = z[end] - width

    pml = zeros(Float32, size(dim))

    for i ∈ axes(pml, 1)
        for j ∈ axes(pml, 2)
            for k ∈ axes(pml, 3)
                depth = maximum([x[i] - start_x, y[j] - start_y, z[k] - start_z, 0.0f0]) / width
                pml[i, j, k] = depth
            end
        end
    end

    return pml .^ 2 * scale
end
