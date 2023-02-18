using ProgressMeter
using IntervalSets

using Waves
using Waves: AbstractDim, AbstractDesign

mutable struct WaveDynamics
    dim::AbstractDim
    grad::AbstractMatrix
    C::SpeedField
    pml::AbstractMatrix
    t::Int
    dt::AbstractFloat
end

function WaveDynamics(;
        dim::AbstractDim, 
        pml_width::Float64, pml_scale::Float64, 
        ambient_speed::Float64, dt::Float64, 
        design::AbstractDesign = nothing)

    grad = gradient(dim.x)

    if !isnothing(design)
        design = DesignInterpolator(design, zero(design), 0.0, 0.0)
    end

    C = SpeedField(dim, ambient_speed, design)
    pml = build_pml(dim, pml_width, pml_scale)

    return WaveDynamics(dim, grad, C, pml, 0, dt)
end

function f(u::AbstractArray, t::Float64, dyn::WaveDynamics)
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
    h = dyn.dt
    t = dyn.t * h

    k1 = f(u, t, dyn)
    k2 = f(u .+ 0.5 * h * k1, t + 0.5 * h, dyn)
    k3 = f(u .+ 0.5 * h * k2, t + 0.5 * h, dyn)
    k4 = f(u .+ h * k3, t + h, dyn)

    return u .+ 1/6 * h * (k1 .+ 2*k2 .+ 2*k3 .+ k4)
end

function integrate(u, dyn::WaveDynamics, n::Int64)
    t = Float64[dyn.t * dyn.dt]
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

# function design_trajectory(design::DesignInterpolator, dt::Float64)
#     sol = typeof(design.initial)[]
#     t = collect(design.ti:dt:design.tf)

#     for i ∈ axes(t, 1)
#         push!(sol, design(t[i]))
#     end

#     return sol
# end

# function design_trajectory(dyn::WaveDynamics)
#     return design_trajectory(dyn.C.design, dyn.dt)
# end

mutable struct WaveEnv
    u::AbstractArray
    dyn::WaveDynamics
    design_steps::Int
end

function design_trajectory(env::WaveEnv)
    design = env.dyn.C.design
    t = collect(range(design.ti, design.tf, env.design_steps + 1))
    traj = typeof(design.initial)[]

    for i ∈ axes(t, 1)
        push!(traj, design(t[i]))
    end

    return traj
end

function Base.time(env::WaveEnv)
    return env.dyn.t * env.dyn.dt
end

function update_design!(env::WaveEnv, action::AbstractDesign)
    ti = time(env)
    tf = ti + env.design_steps * env.dyn.dt
    env.dyn.C.design = DesignInterpolator(env.dyn.C.design(ti), action, ti, tf)
    return nothing
end

function (env::WaveEnv)(action::AbstractDesign)
    update_design!(env, action)
    sol = integrate(env.u, env.dyn, env.design_steps)
    env.u = sol.u[end]
    return sol
end

function is_terminated(env::WaveEnv)
end

function Base.vcat(sol1::WaveSol, sol2::WaveSol)
    pop!(sol1.t)
    pop!(sol1.u)
    return WaveSol(sol1.dim, vcat(sol1.t, sol2.t), vcat(sol1.u, sol2.u))
end

function Base.vcat(sols::WaveSol...)
    return reduce(vcat, sols)
end

kwargs = Dict(:pml_width => 4.0, :pml_scale => 20.0, :ambient_speed => 1.0, :dt => 0.05)
dyn = WaveDynamics(dim = TwoDim(10.0, 0.05), design = Cylinder(-3, -3, 1.0, 0.0); kwargs...)
u = pulse(dyn.dim, 0.0, 0.0, 1.0)
u = cat(u, zeros(size(u)..., 2), dims = 3)
env = WaveEnv(u, dyn, 20)

action_space = Cylinder(-1.0, -1.0, 0.0, 0.0)..Cylinder(1.0, 1.0, 0.0, 0.0)

wave_traj = WaveSol{TwoDim}[]
design_traj = Vector{Cylinder}[]

while env.dyn.t < env.design_steps * 2
    sol = env(rand(action_space))
    
    push!(wave_traj, sol)
    push!(design_traj, design_trajectory(env))
end

wave_traj = vcat(wave_traj...)

designs = Cylinder[]
for i ∈ 1:(length(design_traj))
    pop!(design_traj[i])
    designs = vcat(designs, design_traj)
end

designs = vcat(designs, design_traj[end])

println(length(wave_traj))
println(length(design_traj))