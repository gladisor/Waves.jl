export circle_mask, displacement, energy, flux

"""
Creates a BitMatrix where all elements within the specified radius are 1 and all
else are zero.
"""
function circle_mask(dim::TwoDim, radius::Float32)
    g = grid(dim)
    return dropdims(sum(g .^ 2, dims = 3), dims = 3) .< radius ^2
end

"""
The displacement of a wave is assumed to be its first field.
If the wave is one dimension it will be a matrix with at least two fields, the first
one being displacement. If the wave is two dimensional then it has at least three fields
"""
function displacement(wave::AbstractArray{Float32})
    return selectdim(wave, ndims(wave), 1)
end

"""
Computes the energy of the displacement of the wave.
"""
function energy(u::AbstractArray{Float32})
    return u .^ 2
end

"""
Computes the flux of a scalar field in a particular region given by a mask.
"""
function flux(u::AbstractMatrix{Float32}, laplace::SparseMatrixCSC{Float32}, mask::BitMatrix)
    f = (laplace * u .+ (laplace * u')')
    return sum(f .* mask)
end