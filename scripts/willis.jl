using Waves
using Flux
using ReinforcementLearning
using DataStructures: CircularBuffer

struct WillisDynamics <: AbstractDynamics
    ambient_speed::Float32
    design::DesignInterpolator
    source::AbstractSource
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
        source::AbstractSource = NoSource()
        )

    design = DesignInterpolator(design)
    grid = build_grid(dim)
    grad = build_gradient(dim)
    pml = build_pml(dim, pml_width, pml_scale)
    bc = dirichlet(dim)

    return WillisDynamics(ambient_speed, design, source, grid, grad, pml, bc)
end

function (dyn::WillisDynamics)(wave::AbstractArray{Float32, 3}, t::Float32)
    u_inc_left = wave[:, :, 1]
    v_inc_x_left = 
end

mutable struct WillisEnv <: AbstractEnv
    dim::TwoDim
    reset_wave::AbstractInitialWave
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

dim = TwoDim(20.0f0, 128)
dynamics = WillisDynamics(
    dim, 
    ambient_speed = WATER,
    pml_width = 5.0f0, 
    pml_scale = 10000.0f0)

iter = Integrator(runge_kutta, dynamics, 0.0f0, 1.0f-5, 100)
wave = build_wave(dim, fields = 12)
# dynamics(wave, 0.0f0)

iter(wave)
