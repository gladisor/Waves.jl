import GLMakie
using DifferentialEquations
using DifferentialEquations.OrdinaryDiffEq: ODEIntegrator
using DifferentialEquations: init
using Distributions: Uniform

using Waves
using Waves: AbstractDesign, AbstractDim

"""
Renders an animation of a wave solution.
"""
function render!(sol::WaveSol, design_trajectory::Vector{<:AbstractDesign} = nothing; path::String)
    fig = plot(sol.dim)

    GLMakie.record(fig, path, 1:length(sol)) do i
        GLMakie.empty!(fig.content[1].scene)
        if !isnothing(design_trajectory)
            plot!(fig, design_trajectory[i])
        end
        wave = Wave(sol.dim, sol.u[i])
        plot!(fig, wave)
    end
end

"""
Structure which mediates the interaction between a wave and a changing design.
The effect of the design on the wave occurs through modulation of the wave WaveSpeed
within the media.
"""
mutable struct WaveEnv
    iter::ODEIntegrator
    C::WaveSpeed
    dt::Float64
end

"""
Takes the current environment and updates the WaveSpeed such that an action is applied over
a time interval. It sets the current design equal to the design at the end of the previous time interval.
"""
function update_design!(env::WaveEnv, action)
    design = DesignInterpolator(env.C.design(env.iter.t), action, env.iter.t, env.iter.t + env.dt)
    C = WaveSpeed(env.C.dim, env.C.C0, design)
    env.C = C
    env.iter.p[2] = env.C
end

"""
Propagates the wave simulation to the next time stop which is given by the environment's dt variable.
"""
function propagate_wave!(env::WaveEnv)
    add_tstop!(env.iter, env.iter.t + env.dt)
    step!(env.iter)
end

"""
"""
function interpolate(sol::ODESolution, dim::AbstractDim, dt::Float64)
    u = typeof(sol.prob.u0)[]
    t = collect(sol.prob.tspan[1]:dt:sol.prob.tspan[end])

    for i ∈ axes(t, 1)
        push!(u, sol(t[i]))
    end

    return WaveSol(dim, t, u)
end

function interpolate(designs::Vector{DesignInterpolator}, dt::Float64)
    design_trajectory = typeof(first(designs).initial)[]

    for i ∈ axes(designs, 1)

        design = designs[i]
        t = collect(design.ti:dt:design.tf)

        for j ∈ axes(t, 1)
            push!(design_trajectory, designs[i](t[j]))
        end

        pop!(design_trajectory)
    end

    last_design = last(designs)

    push!(design_trajectory, last_design(last_design.tf))

    return design_trajectory
end

function Waves.plot(x::Vector, y::Vector)
    fig = GLMakie.Figure(resolution = Waves.RESOLUTION, fontsize = Waves.FONT_SIZE)
    ax = GLMakie.Axis(fig[1, 1], title = "Plot", xlabel = "x", ylabel = "y")
    GLMakie.lines!(ax, x, y, color = :blue)
    return fig
end

gs = 10.0
Δ = 0.3
C0 = 2.0
pml_width = 4.0
pml_scale = 10.0
tspan = (0.0, 20.0)
dim = TwoDim(gs, Δ)
u0 = gaussian_pulse(dim, 0.0, 0.0, 1.0)
policy = Uniform.([-2.0, -2.0], [2.0, 2.0])
design = DesignInterpolator(Cylinder(-3, -3, 1.0, 0.2), Cylinder(0.0, 0.0, 0.0, 0.0), tspan...)
C = WaveSpeed(dim, C0, design)
pml = build_pml(dim, pml_width) * pml_scale

prob_tot = ODEProblem(split_wave!, u0, tspan, [Δ, C, pml])
prob_inc = ODEProblem(split_wave!, u0, tspan, [Δ, WaveSpeed(dim, C0), pml])

sol_inc = solve(prob_inc, Midpoint())
iter_tot = init(prob_tot, Midpoint(), advance_to_tstop = true)

env = WaveEnv(iter_tot, C, 1.0)

designs = Vector{DesignInterpolator}()

@time while env.iter.t < tspan[end]
    action = Cylinder(rand.(policy)..., 0.0, 0.0)

    update_design!(env, action)
    propagate_wave!(env)

    println("Time: $(env.iter.t)")
    push!(designs, env.C.design)
end

sol_tot = interpolate(env.iter.sol, dim, 0.1)
sol_inc = interpolate(sol_inc, dim, 0.1)
sol_sc = sol_tot - sol_inc
designs = interpolate(designs, 0.1)
render!(sol_tot, designs, path = "test_tot.mp4")
render!(sol_inc, designs, path = "test_inc.mp4")
render!(sol_sc, designs, path = "test_sc.mp4")

metric = WaveFlux(dim, circle_mask(dim, 6.0))

fig = plot(sol_tot.t, metric(sol_tot))
save(fig, "flux.png")