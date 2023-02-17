using Waves
using Waves: AbstractDim, AbstractDesign

function Base.zero(::Cylinder)
    return Cylinder(0.0, 0.0, 0.0, 0.0)
end

mutable struct WaveDynamics
    dim::AbstractDim
    grad::AbstractMatrix
    C::SpeedField
    pml::AbstractMatrix
    t::AbstractFloat
    dt::AbstractFloat
end

function WaveDynamics(;
        dim::AbstractDim, 
        pml_width::AbstractFloat, pml_scale::AbstractFloat, 
        ambient_speed::AbstractFloat, dt::AbstractFloat, 
        design::AbstractDesign = nothing)

    grad = gradient(dim.x)

    if !isnothing(design)
        design = DesignInterpolator(design, zero(design), 0.0, 0.0)
    end

    C = SpeedField(dim, ambient_speed, design)
    pml = build_pml(dim, pml_width, pml_scale)

    return WaveDynamics(dim, grad, C, pml, 0.0, dt)
end

function f(u::AbstractArray, t::AbstractFloat, dyn::WaveDynamics)
    dt = zeros(size(u))

    U = view(u, :, :, 1)
    Vx = view(u, :, :, 2)
    Vy = view(u, :, :, 3)

    dt[:, :, 1] .= dyn.C(t) .* ((dyn.grad * Vx) .+ (dyn.grad * Vy')') .- U .* dyn.pml
    dt[:, :, 2] .= dyn.grad * U .- Vx .* dyn.pml
    dt[:, :, 3] .= (dyn.grad * U')' .- Vy .* dyn.pml

    return dt
end

function runge_kutta(u::AbstractArray, dyn::WaveDynamics)
    t = dyn.t
    h = dyn.dt

    k1 = f(u, t, dyn)
    k2 = f(u .+ 0.5 * h * k1, t + 0.5 * h, dyn)
    k3 = f(u .+ 0.5 * h * k2, t + 0.5 * h, dyn)
    k4 = f(u .+ h * k3, t + h, dyn)

    return u .+ 1/6 * h * (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end

function integrate(u, dyn::WaveDynamics, tf::AbstractFloat)
    t = typeof(tf)[dyn.t]
    sol = typeof(u)[u]

    while dyn.t < tf
        u = runge_kutta(u, dyn)
        dyn.t += dyn.dt

        push!(t, dyn.t)
        push!(sol, u)
        println(dyn.t)
    end

    return WaveSol(dyn.dim, t, sol)
end

function design_trajectory(dyn::WaveDynamics, tf::AbstractFloat)
    design = dyn.C.design

    sol = typeof(design.initial)[design.initial]
    t = collect(design.ti:dyn.dt:tf)

    for i ∈ axes(t, 1)
        push!(sol, design(t[i]))
    end

    return sol
end

gs = 10.0
dx = 0.05
dt = 0.05
ambient_speed = 1.0
pml_width = 4.0
pml_scale = 20.0
tf = 20.0

dyn = WaveDynamics(
    dim = TwoDim(gs, dx),
    pml_width = pml_width, pml_scale = pml_scale, ambient_speed = ambient_speed, dt = dt,
    design = Cylinder(-3, -3, 1.0, 0.0))

u = pulse(dyn.dim, 0.0, 0.0, 1.0)
u = cat(u, zeros(size(u)..., 2), dims = 3)

@time sol = integrate(u, dyn, tf)
@time design = design_trajectory(dyn, tf)
@time render!(sol, design, path = "vid.mp4")