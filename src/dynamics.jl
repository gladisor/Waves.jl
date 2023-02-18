export WaveDynamics, integrate

"""
Function for constructing a gradient operator for a one dimensional scalar field.
"""
function gradient(x::Vector)
    grad = zeros(size(x, 1), size(x, 1))
    Δ = (x[end] - x[1]) / (length(x) - 1)

    grad[[1, 2, 3], 1] .= [-3.0f0, 4.0f0, -1.0f0] ## left boundary edge case
    grad[[end-2, end-1, end], end] .= [1.0f0, -4.0f0, 3.0f0] ## right boundary edge case

    for i ∈ 2:(size(grad, 2) - 1)
        grad[[i - 1, i + 1], i] .= [-1.0f0, 1.0f0]
    end

    return sparse((grad / (2 * Δ))')
end

mutable struct WaveDynamics
    dim::AbstractDim
    grad::AbstractMatrix
    C::SpeedField
    pml::AbstractMatrix
    t::Int
    dt::Float32
end

function WaveDynamics(;
        dim::AbstractDim, 
        pml_width::Float32, pml_scale::Float32, 
        ambient_speed::Float32, dt::Float32, 
        design::AbstractDesign = nothing)

    grad = gradient(dim.x)

    if !isnothing(design)
        design = DesignInterpolator(design, zero(design), 0.0, 0.0)
    end

    C = SpeedField(dim, ambient_speed, design)
    pml = build_pml(dim, pml_width, pml_scale)

    return WaveDynamics(dim, grad, C, pml, 0, dt)
end

function f(u::AbstractArray, t::Float32, dyn::WaveDynamics)
    dt = zeros(Float32, size(u))

    U = view(u, :, :, 1)
    Vx = view(u, :, :, 2)
    Vy = view(u, :, :, 3)

    dt[:, :, 1] .= dyn.C(t) .* ((dyn.grad * Vx) .+ (dyn.grad * Vy')') .- U .* dyn.pml
    dt[:, :, 2] .= dyn.grad * U .- Vx .* dyn.pml
    dt[:, :, 3] .= (dyn.grad * U')' .- Vy .* dyn.pml

    return dt
end

function runge_kutta(u::AbstractArray, dyn::WaveDynamics)
    h = dyn.dt
    t = dyn.t * h

    k1 = f(u, t, dyn)
    k2 = f(u .+ 0.5f0 * h * k1, t + 0.5f0 * h, dyn)
    k3 = f(u .+ 0.5f0 * h * k2, t + 0.5f0 * h, dyn)
    k4 = f(u .+ h * k3, t + h, dyn)

    return u .+ 1/6f0 * h * (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end

function integrate(u, dyn::WaveDynamics, n::Int64)
    t = Float32[dyn.t * dyn.dt]
    sol = typeof(u)[u]
    tf = dyn.t + n

    while dyn.t < tf
        u = runge_kutta(u, dyn)
        dyn.t += 1

        push!(t, dyn.t * dyn.dt)
        push!(sol, u)
    end

    return WaveSol(dyn.dim, t, sol)
end