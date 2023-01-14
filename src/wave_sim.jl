export WaveSim, get_data, render!, reset!, WaveSol

mutable struct WaveSim{D <: AbstractDim}
    sys::PDESystem
    disc::MOLFiniteDifference
    grid::Dict
    prob::ODEProblem
    iter::ODEIntegrator
    wave::Wave{D}
    dt::Real
end

function WaveSim(;
        wave::Wave, 
        ic::InitialCondition, 
        t_max::Real, 
        speed::Real, 
        n::Int, 
        dt::Real,
        design::Union{ParameterizedDesign, Nothing} = nothing)

    ps = [wave.speed => speed]

    if isnothing(design)
        eq = wave_equation(wave)
    else
        eq = wave_equation(wave, design)
        ps = vcat(ps, design_parameters(design, design.design, 0.0, t_max))
    end

    @named sys = PDESystem(
        eq, conditions(wave, ic), get_domain(wave, t_max = t_max),
        spacetime(wave), [signature(wave)], ps)

    disc = wave_discretizer(wave, n)
    prob = discretize(sys, disc)
    grid = get_discrete(sys, disc)
    iter = init(prob, Tsit5(), advance_to_tstop = true, saveat = dt)
    return WaveSim(sys, disc, grid, prob, iter, wave, dt)
end

function Waves.step!(sim::WaveSim, dt::Real)
    add_tstop!(sim.iter, sim.iter.t + dt)
    step!(sim.iter)
end

function Waves.step!(sim::WaveSim)
    step!(sim.iter)
end

function Base.display(sim::WaveSim)
    display(typeof(sim))
end

function reset!(sim::WaveSim)
    reinit!(sim.iter)
    return nothing
end

function get_data(sim::WaveSim)
    return sim.iter.sol[sim.grid[signature(sim.wave)]]
end

function state(sim::WaveSim)
    return sim.iter[sim.grid[Waves.signature(sim.wave)]]
end

function dims(sim::WaveSim)::Vector
    return [collect(sim.grid[d]) for d ∈ dims(sim.wave)]
end

struct WaveSol{D <: AbstractDim}
    wave::Wave{D}
    # grid::Dict
    # sol::ODESolution
    dims::Vector
    data::AbstractArray
end

# function WaveSol(sim::WaveSim)
#     return WaveSol(sim.wave, sim.grid, sim.iter.sol)
# end

function WaveSol(sim::WaveSim)
    return WaveSol(sim.wave, dims(sim), get_data(sim))
end

function Base.display(sol::WaveSol)
    display(typeof(sol))
end

function Base.:-(sol::WaveSol, other::WaveSol)
    data = sol.data .- other.data
    return WaveSol(sol.wave, sol.dims, data)
end

# function get_data(sol::WaveSol)
#     return sol.sol[sol.grid[signature(sol.wave)]]
# end

# function dims(s::Union{WaveSim, WaveSol})::Vector
#     return [collect(s.grid[d]) for d ∈ dims(s.wave)]
# end

function render!(sol::WaveSol{OneDim}; path::String)
    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
    ax = GLMakie.Axis(fig[1, 1], title = "1D Wave", xlabel = "X", ylabel = "Y")

    GLMakie.xlims!(ax, getbounds(sol.wave.dim.x)...)
    GLMakie.ylims!(ax, -1.0, 1.0)

    x = sol.dims[1]
    data = sol.data

    GLMakie.record(fig, path, axes(data, 1)) do i
        GLMakie.empty!(ax.scene)
        GLMakie.lines!(ax, x, data[i], linestyle = nothing, linewidth = 5, color = :blue)
    end

    return nothing
end

function Waves.render!(
        sol::WaveSol{TwoDim}; 
        path::String, 
        design::Union{Vector{<: AbstractDesign}, Nothing} = nothing)
        
    fig = GLMakie.Figure(resolution = (1920, 1080), fontsize = 20)
    ax = GLMakie.Axis3(fig[1,1], aspect = (1, 1, 1), perspectiveness = 0.5, title = "2D Wave", xlabel = "X", ylabel = "Y", zlabel = "Z")

    GLMakie.xlims!(ax, getbounds(sol.wave.dim.x)...)
    GLMakie.ylims!(ax, getbounds(sol.wave.dim.y)...)
    GLMakie.zlims!(ax, 0.0, 5.0)

    x, y = sol.dims
    data = sol.data

    GLMakie.record(fig, path, axes(data, 1)) do i
        GLMakie.empty!(ax.scene)
        GLMakie.surface!(ax, x, y, data[i], colormap = :ice, shading = false)
        if !isnothing(design)
            GLMakie.mesh!(ax, design[i])
        end
    end

    return nothing
end