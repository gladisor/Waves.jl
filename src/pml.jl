export build_pml

"""
Creates a pml for a one dimensional wave.
"""
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
Creates a pml for the x direction of a two dimentional wave. If the axes are symetric
then this pml should work for the y direction if it is transposed.
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