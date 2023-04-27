include("dependencies.jl")

abstract type AbstractSource end

struct NoSource <: AbstractSource end
(source::NoSource)(t::Float32) = 0.0f0

struct Source <: AbstractSource
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
    source::AbstractSource
    grid::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor WaveDynamics
Flux.trainable(dyn::WaveDynamics) = (;)

function WaveDynamics(dim::AbstractDim; 
        ambient_speed::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        design::AbstractDesign = NoDesign(),
        source::AbstractSource = NoSource()
        )

    design = DesignInterpolator(design)
    grid = build_grid(dim)
    grad = build_gradient(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    bc = dirichlet(dim)

    return WaveDynamics(ambient_speed, design, source, grid, grad, pml, bc)
end

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
    Ux = ∂x(∇, U .+ force)
    Uy = ∂y(∇, U .+ force)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U

    return cat(dyn.bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function update_design(dyn::WaveDynamics, tspan::Vector{Float32}, action::AbstractDesign)
    initial = dyn.design(tspan[1])
    design = DesignInterpolator(initial, action, tspan[1], tspan[end])
    return WaveDynamics(dyn.ambient_speed, design, dyn.source, dyn.grid, dyn.grad, dyn.pml, dyn.bc)
end

dim = TwoDim(8.0f0, 256)
wave = build_wave(dim, fields = 6) |> gpu
pulse = build_pulse(build_grid(dim), 0.0f0, -3.0f0, 5.0f0)

dynamics = WaveDynamics(
    dim,
    ambient_speed = AIR,
    pml_width = 2.0f0,
    pml_scale = 20000.0f0,
    design = Scatterers([0.0f0 0.0f0], [1.0f0], [BRASS]),
    source = Source(pulse, 300.0f0))

iter = Integrator(runge_kutta, dynamics, 0.0f0, 5e-5, 500) |> gpu

tspan = build_tspan(iter)
@time u = iter(wave) |> cpu

u = linear_interpolation(tspan, unbatch(u))
@time render!(dim, tspan, u, path = "vid.mp4", seconds = 5.0f0)