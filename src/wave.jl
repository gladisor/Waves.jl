export Wave

struct Wave{D <: AbstractDim}
    dim::D
    speed::Num
    t::Num
    u::CallWithMetadata
end

function Wave(;dim)
    @parameters speed
    @parameters t
    @variables u(..)
    return Wave(dim, speed, t, u)
end

function dims(wave::Wave)::Tuple
    return dims(wave.dim)
end

function wave_equation(wave::Wave{TwoDim}, C::Function)::Equation
    x, y = dims(wave)
    t, u = wave.t, wave.u

    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dtt = Differential(t)^2

    return Dtt(u(x, y, t)) ~ C(x, y, t)^2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))
end

function wave_equation(wave::Wave{TwoDim})::Equation
    return wave_equation(wave, (x, y, t) -> wave.speed)
end

function dirichlet(wave::Wave{TwoDim})::Vector{Equation}
    x, y = dims(wave)
    t, u = wave.t, wave.u

    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)

    return [
        u(x_min, y, t) ~ 0., 
        u(x_max, y, t) ~ 0.,
        u(x, y_min, t) ~ 0.,
        u(x, y_max, t) ~ 0.]
end

function neumann(wave::Wave{TwoDim})::Vector{Equation}
    x, y = dims(wave)
    t, u = wave.t, wave.u

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

function time_condition(wave::Wave{TwoDim})::Equation
    x, y = dims(wave)
    t, u = wave.t, wave.u
    Dt = Differential(t)
    return Dt(u(x, y, 0.0)) ~ 0.
end

function boundary_conditions(wave::Wave)::Vector{Equation}
    return vcat(dirichlet(wave), neumann(wave), [time_condition(wave)])
end

function get_domain(wave::Wave; t_max)
    domain = []

    for d ∈ dims(wave)
        push!(domain, d ∈ getbounds(d))
    end

    push!(domain, wave.t ∈ (0.0, t_max))
    return domain
end

function (wave::Wave)(;ic::Function, speed::Real, name)

    bcs = [
        wave.u(dims(wave)..., 0.0) ~ ic(dims(wave)...)
        boundary_conditions(wave)]

    sys = PDESystem(
        wave_equation(wave), 
        bcs, 
        get_domain(wave, t_max = 10.0), 
        [dims(wave), wave.t], 
        [wave.u(dims(wave)..., wave.t)],
        [wave.speed => speed]
        ; name = name)

    return sys
end