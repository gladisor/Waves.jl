export Wave2D

struct Wave2D <: AbstractWave
    x::Num
    y::Num
    t::Num
    u::CallWithMetadata
end

function Wave2D(;x_min, x_max, y_min, y_max, t_max)
    @parameters x [bounds = (x_min, x_max)]
    @parameters y [bounds = (y_min, y_max)]
    @parameters t [bounds = (0.0, t_max)]
    @variables u(..)

    return Wave2D(x, y, t, u)
end

function wave_equation(wave::Wave2D, C::Function)
    x, y, t, u = wave.x, wave.y, wave.t, wave.u
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dtt = Differential(t)^2

    return Dtt(u(x, y, t)) ~ C(x, y, t)^2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))
end

function dirichlet(wave::Wave2D)
    x, y, t, u = wave.x, wave.y, wave.t, wave.u
    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)

    return [
        u(x_min, y, t) ~ 0., 
        u(x_max, y, t) ~ 0.,
        u(x, y_min, t) ~ 0.,
        u(x, y_max, t) ~ 0.]
end

function neumann(wave::Wave2D)
    x, y, t, u = wave.x, wave.y, wave.t, wave.u
    Dx = Differential(x)
    Dy = Differential(y)
    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)

    return [
        Dx(u(x_min, y, t)) ~ 0., 
        Dx(u(x_max, y, t)) ~ 0.,
        Dy(u(x, y_min, t)) ~ 0.,
        Dy(u(x, y_max, t)) ~ 0.
        ]
end

function time_condition(wave::Wave2D)::Equation
    x, y, t, u = wave.x, wave.y, wave.t, wave.u
    Dt = Differential(t)
    t_min, _ = getbounds(wave.t)

    return Dt(u(x, y, t_min)) ~ 0.
end

function boundary_conditions(wave::AbstractWave)::Vector{Equation}
    return vcat(dirichlet(wave), neumann(wave), [time_condition(wave)])
end

struct WaveSolution2D
    x::Vector
    y::Vector
    t::Vector
    u::Array
end

function animate!(sol::WaveSolution2D, path::String; zlim = (-1.0, 1.0))

    x, y, t, u = sol.x, sol.y, sol.t, sol.u
    a = Animation()
    @showprogress for i ∈ axes(u, 3)
        p = surface(
            x, y, u[:, :, i],
            title = "Wave at T = u(x, $(t[i]))",
            xlabel = "x",
            ylabel = "y",
            zlim = zlim,
            legend = false,
            size=(1920, 1080))
        
        frame(a, p)
    end

    gif(a, path, fps = 20)
end