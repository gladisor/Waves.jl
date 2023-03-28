export WaveDynamics

"""
Structure for holding information on how the wave should change over time.

    dim: holds information on the dimentional space
    g: constructed from dim, is the array of coordinate points
    grad: used to take first derivatives of scalar fields
    design: holds a structure which has an effect on the wave
    pml: the rate at which outgoing waves should be dampened
    ambient_speed: the normal speed of the wave as it moves through the domain
    t: integer timestep
    dt: the amount of time that passes at each timestep
"""
mutable struct WaveDynamics <: AbstractDynamics
    dim::AbstractDim
    g::AbstractArray{Float32}
    grad::AbstractMatrix{Float32}
    design::Union{DesignInterpolator, Nothing}
    pml::AbstractArray{Float32}
    ambient_speed::Float32
    t::Int
    dt::Float32
end

Flux.@functor WaveDynamics
Flux.trainable(dynamics::WaveDynamics) = ()

function WaveDynamics(;
        dim::AbstractDim, 
        pml_width::Float32, 
        pml_scale::Float32, 
        ambient_speed::Float32, 
        dt::Float32,
        design::Union{AbstractDesign, Nothing} = nothing,
        )

    g = grid(dim)
    grad = gradient(dim.x)
    design = DesignInterpolator(design)
    pml = build_pml(dim, pml_width, pml_scale)

    return WaveDynamics(dim, g, grad, design, pml, ambient_speed, 0, dt)
end


"""
For resetting the time of the dynamics
"""
function reset!(dyn::WaveDynamics)
    dyn.t = 0
end

"""
Gets the physical time in the system
"""
function Base.time(dyn::WaveDynamics)
    return dyn.t * dyn.dt
end

"""
Retrives the wave speed array which describes the speed of the wave at every point
within the domain. If there is no design then the speed is just the ambient_speed everywhere.
If there is a design then the speed of the design is included where the design exists.
"""
function speed(dyn::WaveDynamics, t::Float32)
    if isnothing(dyn.design)
        return one(dyn.dim) * dyn.ambient_speed
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

function DesignTrajectory(dyn::WaveDynamics, n::Int)
    design = dyn.design
    t = collect(range(design.ti, design.tf, n + 1))
    traj = typeof(design.initial)[]

    for i ∈ axes(t, 1)
        push!(traj, design(t[i]))
    end

    return DesignTrajectory(traj)
end