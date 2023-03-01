export WaveDynamics

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

function reset!(dyn::WaveDynamics)
    dyn.t = 0
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