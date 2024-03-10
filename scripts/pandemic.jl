using Waves, Flux, CairoMakie

Flux.device!(0)
Flux.CUDA.allowscalar(false)

struct PandemicDynamics <: AbstractDynamics
    grad::AbstractMatrix{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor PandemicDynamics
Flux.trainable(dyn::PandemicDynamics) = (;)

function (dyn::PandemicDynamics)(x::AbstractArray{Float32, 3}, t::AbstractVector{Float32}, θ::Vector)
    F = θ[1]
    f = F(t)

    U = x[:, :, 1]
    Vx = x[:, :, 2]
    Vy = x[:, :, 3]

    Vxx = ∂x(dyn.grad, Vx)
    Vyy = ∂y(dyn.grad, Vy)
    Ux = ∂x(dyn.grad, U .+ f)
    Uy = ∂y(dyn.grad, U .+ f)
    
    dU = WATER * (Vxx .+ Vyy)
    dVx = WATER * Ux
    dVy = WATER * Uy
    return cat(dU .* dyn.bc, dVx, dVy, dims = 3)
end

dim = TwoDim(5.0f0, 512)
grid = build_grid(dim)

dyn = PandemicDynamics(build_gradient(dim), build_dirichlet(dim))
iter = gpu(Integrator(runge_kutta, dyn, 1f-5))
t = build_tspan(iter, 0.0f0, 1000)

design_interp = gpu(DesignInterpolator(NoDesign(), NoDesign(), t[1], t[end]))

source_shape = build_normal(grid, [0.0f0 0.0f0], [0.3f0], [1.0f0])
F = gpu(Source(source_shape, 1000.0f0))

grid = gpu(grid)
t = gpu(t)
ic = gpu(build_wave(dim, 3))
sol = cpu(iter(ic, t[:, :], [F]))

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0)
record(fig, "pandemic.mp4", axes(t, 1)) do i
    empty!(ax)
    heatmap!(ax, dim.x, dim.y, sol[:, :, 1, i], colorrange = (-1.5f0, 1.5f0))
end
