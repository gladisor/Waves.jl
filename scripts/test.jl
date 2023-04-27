include("dependencies.jl")

struct Source
    source::AbstractArray{Float32}
    freq::Float32
end

Flux.@functor Source

function (source::Source)(t::Float32)
    return source.source * sin(2.0f0 * pi * t * source.freq)
end

struct WaveDynamics <: AbstractDynamics
    ambient_speed::Float32
    design::DesignInterpolator
    source::Source
    grid::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor WaveDynamics
Flux.trainable(dyn::WaveDynamics) = (;)

function (dyn::WaveDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    C = speed(dyn.design(t), dyn.grid, dyn.ambient_speed)
    b = C .^ 2
    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'
    force = dyn.source(t)

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U)
    Uy = ∂y(∇, U)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx .+ force
    dVy = Uy .- σy .* Vy .+ force
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dyn.bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

dim = TwoDim(8.0f0, 256)
wave = build_wave(dim, fields = 6) |> gpu
grid = build_grid(dim)
grad = build_gradient(dim)
pml = build_pml(dim, 2.0f0, 20000.0f0)
bc = dirichlet(dim)

design = DesignInterpolator(NoDesign())
source = Source(build_pulse(grid, 0.0f0, 0.0f0, 1.0f0), 200.0f0)

dynamics = WaveDynamics(AIR, design, source, grid, grad, pml, bc)

iter1 = Integrator(runge_kutta, dynamics, 0.0f0, 5e-5, 500) |> gpu

ti = iter1.ti + iter1.steps * iter1.dt
iter2 = Integrator(runge_kutta, dynamics, ti, 5e-5, 500) |> gpu

@time u1 = iter1(wave) |> cpu
tspan1 = build_tspan(iter1)
@time u2 = iter2(gpu(u1[:, :, :, end])) |> cpu
tspan2 = build_tspan(iter2)

tspan = vcat(tspan1, tspan2)
u = cat(u1, u2, dims = ndims(u1))

u = linear_interpolation(tspan, unbatch(u))
@time render!(dim, tspan, u, path = "vid.mp4", seconds = 10.0f0)