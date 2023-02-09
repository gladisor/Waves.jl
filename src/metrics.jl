abstract type WaveMetric end

struct WaveFlux <: WaveMetric
    Δ::Float64
    mask::AbstractArray{Float64}
end

function WaveFlux(dim::AbstractDim, mask = nothing)
    Δ = (dim.x[end] - dim.x[1]) / (length(dim.x) - 1)
    mask = isnothing(mask) ? ones(size(dim)) : mask
    return WaveFlux(Δ, mask)
end

"""
Compute flux on the state of a one dimensional wave simulation.
"""
function (metric::WaveFlux)(u::Vector{Float64})
    return sum(∇(∇(u, metric.Δ), metric.Δ) .* metric.mask)
end

function (metric::WaveFlux)(u::Matrix{Float64})
    d = ∇x(∇x(u, metric.Δ), metric.Δ) .+ ∇y(∇y(u, metric.Δ), metric.Δ)
    return sum(d .* metric.mask)
end

function (metric::WaveFlux)(sol::WaveSol{OneDim})
    return [metric(u[:, 1]) for u ∈ sol.u]
end

function (metric::WaveFlux)(sol::WaveSol{TwoDim})
    return [metric(u[:, :, 1]) for u ∈ sol.u]
end

function square_mask(dim::OneDim, radius::Float64)
    mask = zeros(size(dim))
    mask[(-radius .< dim.x .< radius)] .= 1.0
    return mask
end

function square_mask(dim::TwoDim, radius::Float64)
    mask = zeros(size(dim))
    mask[(-radius .< dim.x .< radius), (-radius .< dim.y .< radius)] .= 1.0
    return mask
end

function circle_mask(dim::TwoDim, radius::Float64)
    mask = zeros(size(dim))
    g = grid(dim)

    for i ∈ axes(mask, 1)
        for j ∈ axes(mask, 2)
            x, y = g[i, j]
            mask[i, j] = (x ^ 2 + y ^ 2) <= radius ^ 2
        end
    end

    return mask
end