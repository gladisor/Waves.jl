using Waves
using Flux
using ReinforcementLearning
using DataStructures: CircularBuffer
using CairoMakie

struct WillisDynamics <: AbstractDynamics
    ambient_speed::Float32
    design::DesignInterpolator
    source¹::AbstractSource
    source²::AbstractSource
    grid::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    pml::AbstractArray{Float32}
    bc::AbstractArray{Float32}
end

Flux.@functor WillisDynamics
Flux.trainable(::WillisDynamics) = (;)

function WillisDynamics(dim::AbstractDim; 
        ambient_speed::Float32,
        pml_width::Float32,
        pml_scale::Float32,
        design::AbstractDesign = NoDesign(),
        source¹::AbstractSource = NoSource(),
        source²::AbstractSource = NoSource()
        )

    design = DesignInterpolator(design)
    grid = build_grid(dim)
    grad = build_gradient(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    bc = dirichlet(dim)

    return WillisDynamics(ambient_speed, design, source¹, source², grid, grad, pml, bc)
end

function pml_wave_dynamics(wave, b, ∇, σx, σy, f, bc)

    U = wave[:, :, 1]
    Vx = wave[:, :, 2]
    Vy = wave[:, :, 3]
    Ψx = wave[:, :, 4]
    Ψy = wave[:, :, 5]
    Ω = wave[:, :, 6]

    Vxx = ∂x(∇, Vx)
    Vyy = ∂y(∇, Vy)
    Ux = ∂x(∇, U .+ f)
    Uy = ∂y(∇, U .+ f)

    dU = b .* (Vxx .+ Vyy) .+ Ψx .+ Ψy .- (σx .+ σy) .* U .- Ω
    dVx = Ux .- σx .* Vx
    dVy = Uy .- σy .* Vy
    dΨx = b .* σx .* Vyy
    dΨy = b .* σy .* Vxx
    dΩ = σx .* σy .* U
    return cat(bc .* dU, dVx, dVy, dΨx, dΨy, dΩ, dims = 3)
end

function (dyn::WillisDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    wave¹_inc = wave[:, :, 1:6]
    wave¹_tot = wave[:, :, 7:12]
    wave²_inc = wave[:, :, 13:18]
    wave²_tot = wave[:, :, 19:24]

    C = speed(dyn.design(t), dyn.grid, dyn.ambient_speed)
    b = C .^ 2
    b0 = dyn.ambient_speed ^ 2

    ∇ = dyn.grad
    σx = dyn.pml
    σy = σx'
    f¹ = dyn.source¹(t)
    f² = dyn.source²(t)

    dwave¹_inc = pml_wave_dynamics(wave¹_inc, b0, ∇, σx, σy, f¹, dyn.bc)
    dwave¹_tot = pml_wave_dynamics(wave¹_tot, b, ∇, σx, σy, f¹, dyn.bc)
    dwave²_inc = pml_wave_dynamics(wave²_inc, b0, ∇, σx, σy, f², dyn.bc)
    dwave²_tot = pml_wave_dynamics(wave²_tot, b, ∇, σx, σy, f², dyn.bc)

    return cat(dwave¹_inc, dwave¹_tot, dwave²_inc, dwave²_tot, dims = 3)
end

mutable struct WillisEnv <: AbstractEnv
    dim::TwoDim
    design_space::DesignSpace
    action_speed::Float32

    wave::CircularBuffer{AbstractArray{Float32, 3}}
    dynamics::WillisDynamics

    image_resolution::Tuple{Int, Int}
    
    signal::Vector{Float32}
    time_step::Int
    integration_steps::Int
    actions::Int
end

Flux.@functor WillisEnv
Flux.trainable(::WillisEnv) = (;)

dim = TwoDim(20.0f0, 64)
grid = build_grid(dim)
pulse¹ = exp.(- 2.0f0 * (grid[:, :, 1] .- -15.0f0) .^ 2)
pulse² = exp.(- 2.0f0 * (grid[:, :, 1] .- 15.0f0) .^ 2)

source¹ = Source(pulse¹, freq = 500.0f0)
source² = Source(pulse², freq = 500.0f0)

rot = Float32.(Waves.build_2d_rotation_matrix(30))
pos = vcat(
        [0.0f0, 0.0f0]',
        Waves.hexagon_ring(3.5f0),
        Waves.hexagon_ring(4.75f0) * rot,
        Waves.hexagon_ring(6.0f0)
    )

DESIGN_SPEED = 3 * AIR

r_low = fill(0.2f0, size(pos, 1))
r_high = fill(1.0f0, size(pos, 1))
c = fill(DESIGN_SPEED, size(pos, 1))

design_low = AdjustableRadiiScatterers(Cylinders(pos, r_low, c))
design_high = AdjustableRadiiScatterers(Cylinders(pos, r_high, c))
ds = DesignSpace(design_low, design_high)

dynamics = WillisDynamics(
    dim, 
    ambient_speed = WATER,
    pml_width = 5.0f0, 
    pml_scale = 10000.0f0,
    source¹ = source¹,
    source² = source²,
    design = rand(ds))

iter = Integrator(runge_kutta, dynamics, 0.0f0, 1.0f-5, 1000)
wave = build_wave(dim, fields = 24)
@time u = iter(wave)

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0f0)
heatmap!(ax1, dim.x, dim.y, u[:, :, 7, end], colormap = :ice)

ax2 = Axis(fig[1, 2], aspect = 1.0f0)
heatmap!(ax2, dim.x, dim.y, u[:, :, 19, end], colormap = :ice)
save("u.png", fig)