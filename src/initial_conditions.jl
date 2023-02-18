export pulse

function pulse(dim::OneDim, x = 0.0, intensity = 1.0)
    u = exp.(- intensity * (dim.x .- x) .^ 2)
    return cat(u, zeros(size(u)), dims = 2)
end

function pulse(dim::TwoDim, x = 0.0, y = 0.0, intensity = 1.0)
    u = zeros(length(dim.x), length(dim.y))

    for i ∈ axes(u, 1)
        for j ∈ axes(u, 2)
            u[i, j] = exp(- intensity * ((dim.x[i] - x) .^ 2 + (dim.y[j] - y) .^ 2))
        end
    end

    return cat(u, zeros(size(u)..., 2), dims = 3)
end