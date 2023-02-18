export pulse

function pulse(dim::OneDim, x::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    u = exp.(- intensity * (dim.x .- x) .^ 2)
    return cat(u, zeros(Float32, size(u)), dims = 2)
end

function pulse(dim::TwoDim, x::Float32 = 0.0f0, y::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    u = zeros(Float32, length(dim.x), length(dim.y))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            u[i, j] = exp(- intensity * ((dim.x[i] - x) .^ 2 + (dim.y[j] - y) .^ 2))
        end
    end

    return cat(u, zeros(Float32, size(u)..., 2), dims = 3)
end