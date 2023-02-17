using Waves
using Waves: AbstractDim

gs = 10.0
dx = 0.05
dt = 0.05
ambient_speed = 1.0
pml_width = 4.0
pml_scale = 20.0
tspan = (0.0, 20.0)

dim = TwoDim(gs, dx)
u = pulse(dim, 0.0, 0.0, 5.0)
u0 = cat(u, zeros(size(u)..., 2), dims = 3)
grad = gradient(dim.x)
design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.1), Cylinder(6.0, 6.0, 0.0, 0.0), tspan...)
C = SpeedField(dim, 1.0, design)
pml = build_pml(dim, pml_width, pml_scale)

mutable struct WaveDynamics
    dim::AbstractDim
    grad::Matrix
    C::SpeedField
    pml::Matrix
    t::AbstractFloat
    dt::AbstractFloat
end

dyn = WaveDynamics(dim, grad, C, pml, 0.0, dt)

function f(u, dyn::WaveDynamics)
    dt = zeros(size(u))

    U = view(u, :, :, 1)
    Vx = view(u, :, :, 2)
    Vy = view(u, :, :, 3)

    dt[:, :, 1] .= dyn.C(dyn.t) .* ((dyn.grad * Vx) .+ (dyn.grad * Vy')') .- U .* dyn.pml
    dt[:, :, 2] .= dyn.grad * U .- Vx .* dyn.pml
    dt[:, :, 3] .= (dyn.grad * U')' .- Vy .* dyn.pml

    return dt
end

function runge_kutta(u, dyn::WaveDynamics)
    h = dyn.dt

    k1 = f(u, dyn)
    k2 = f(u .+ 0.5 * h * k1, dyn)
    k3 = f(u .+ 0.5 * h * k2, dyn)
    k4 = f(u .+ h * k3, dyn)

    return u .+ 1/6 * h * (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end

function integrate(u, dyn::WaveDynamics)
    t = Float64[dyn.t]
    sol = typeof(u)[u]

    while dyn.t < tspan[end]
        u = runge_kutta(u, dyn)
        dyn.t += dyn.dt

        push!(t, dyn.t)
        push!(sol, u)
        println(dyn.t)
    end

    return WaveSol(dyn.dim, t, sol)
end

@time sol = integrate(u0, dyn)
@time render!(sol, path = "vid.mp4")




