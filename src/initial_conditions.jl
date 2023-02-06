export gaussian_pulse

function gaussian_pulse(dim::OneDim, x, intensity)
    u = zeros(size(dim))

    for i ∈ axes(u, 1)
        u[i] = exp(-intensity * (dim.x[i] - x) ^ 2)
    end

    v = zeros(size(u))
    u0 = cat(u, v, dims = (ndims(u) + 1))
    return u0
end

function gaussian_pulse(dim::TwoDim, x, y, intensity)

    u = zeros(size(dim))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            u[i, j] = exp(- intensity * ((dim.x[i] - x) ^ 2 + (dim.y[j] - y) ^ 2))
        end
    end

    v = zeros(size(u)..., 2)
    u0 = cat(u, v, dims = (ndims(u) + 1))
    return u0
end

function gaussian_pulse(dim::ThreeDim, x, y, z, intensity)
    u = zeros(size(dim))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            for k ∈ axes(u, 3)
                u[i, j, k] = exp(- intensity * ((dim.x[i] - x) ^ 2 + (dim.y[j] - y) ^ 2 + (dim.z[k] - z) ^ 2))
            end
        end
    end

    v = zeros(size(u)..., 3)
    u0 = cat(u, v, dims = (ndims(u) + 1))
    return u0
end