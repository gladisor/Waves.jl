export circle_mask, displacement, energy, flux

function circle_mask(dim::TwoDim, radius::Float32)
    g = grid(dim)
    return dropdims(sum(g .^ 2, dims = 3), dims = 3) .< radius ^2
end

function displacement(wave::AbstractArray{Float32})
    return selectdim(wave, ndims(wave), 1)
end

function energy(u::AbstractArray{Float32})
    return u .^ 2
end

function flux(u::AbstractMatrix{Float32}, laplace::SparseMatrixCSC{Float32}, mask::BitMatrix)
    f = (laplace * u .+ (laplace * u')')
    return sum(f .* mask)
end