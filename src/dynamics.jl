export WaveDynamics

const FORWARD_DIFF_COEF = [-3.0f0, 4.0f0, -1.0f0]
const BACKWARD_DIFF_COEF = [1.0f0, -4.0f0, 3.0f0]
const CENTRAL_DIFF_COEF = [-1.0f0, 1.0f0]

"""
Function for constructing a gradient operator for a one dimensional scalar field.
"""
function gradient(x::Vector{Float32})
    grad = zeros(Float32, size(x, 1), size(x, 1))
    Δ = (x[end] - x[1]) / (length(x) - 1)

    grad[[1, 2, 3], 1] .= FORWARD_DIFF_COEF ## left boundary edge case
    grad[[end-2, end-1, end], end] .= BACKWARD_DIFF_COEF ## right boundary edge case

    for i ∈ 2:(size(grad, 2) - 1)
        grad[[i - 1, i + 1], i] .= CENTRAL_DIFF_COEF
    end

    return sparse((grad / (2 * Δ))')
end

function stable_dt(dx::Float32, ambient_speed::Float32)::Float32
    return sqrt(dx^2 / ambient_speed^2)
end

mutable struct WaveDynamics
    dim::AbstractDim
    g::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    design::Union{DesignInterpolator, Nothing}
    pml::AbstractArray{Float32}
    ambient_speed::Float32
    t::Int
    dt::Float32
end

function WaveDynamics(;
        dim::AbstractDim, design::Union{AbstractDesign, Nothing} = nothing,
        pml_width::Float32, pml_scale::Float32, 
        ambient_speed::Float32, dt::Float32)

    g = grid(dim)
    grad = gradient(dim.x)
    design = DesignInterpolator(design)
    pml = build_pml(dim, pml_width, pml_scale)

    return WaveDynamics(dim, g, grad, design, pml, ambient_speed, 0, dt)
end

function Base.time(dyn::WaveDynamics)
    return dyn.t * dyn.dt
end

function speed(dyn::WaveDynamics, t::Float32)
    if isnothing(dyn.design)
        return dropdims(sum(dyn.g, dims = 3), dims = 3) .^ 0.0f0 * dyn.ambient_speed
    else
        return speed(dyn.design(t), dyn.g, dyn.ambient_speed)
    end
end

function Flux.gpu(dyn::WaveDynamics)
    return WaveDynamics(
        gpu(dyn.dim), 
        gpu(dyn.g),
        gpu(dyn.grad), 
        gpu(dyn.design),
        gpu(dyn.pml), 
        dyn.ambient_speed, dyn.t, dyn.dt)
end

function Flux.cpu(dyn::WaveDynamics)
    return WaveDynamics(
        cpu(dyn.dim),
        cpu(dyn.g),
        cpu(dyn.grad),
        cpu(dyn.design),
        cpu(dyn.pml),
        dyn.ambient_speed, dyn.t, dyn.dt)
end