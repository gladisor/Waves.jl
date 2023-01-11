export Wave, wave_equation, wave_speed, boundary_conditions, get_domain

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

function unpack(wave::Wave)
    return (dims(wave)..., wave.t, wave.u)
end

function wave_equation(wave::Wave{OneDim}, C::Function)::Equation
    x, t, u = unpack(wave)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    return Dtt(u(x, t)) ~ C(x, t) ^ 2 * Dxx(u(x, t))
end

function wave_equation(wave::Wave{TwoDim}, C::Function)::Equation
    x, y, t, u = unpack(wave)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dtt = Differential(t)^2
    return Dtt(u(x, y, t)) ~ C(x, y, t)^2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))
end

function wave_speed(wave::Wave{OneDim})::Function
    return (x, t) -> wave.speed
end

function wave_speed(wave::Wave{TwoDim})::Function
    return (x, y, t) -> wave.speed
end

function wave_speed(wave::Wave{ThreeDim})::Function
    return (x, y, z, t) -> wave.speed
end

"""
When there is no design present in the spacial domain of the wave then
the wave speed scalar function takes on the default value of the wave.
"""
function Waves.wave_equation(wave::Wave)::Equation
    return wave_equation(wave, wave_speed(wave))
end

function dirichlet(wave::Wave{OneDim})::Vector{Equation}
    x, t, u = unpack(wave)
    x_min, x_max = getbounds(x)

    return [
        u(x_min, t) ~ 0.,
        u(x_max, t) ~ 0.]
end

"""
The dirichlet boundary condition for two dimensions which specifies that the displacement
of the wave along the boundary of the spacial domain is always zero.
"""
function dirichlet(wave::Wave{TwoDim})::Vector{Equation}
    x, y, t, u = unpack(wave)
    x_min, x_max = getbounds(x)
    y_min, y_max = getbounds(y)

    return [
        u(x_min, y, t) ~ 0., 
        u(x_max, y, t) ~ 0.,
        u(x, y_min, t) ~ 0.,
        u(x, y_max, t) ~ 0.]
end

function neumann(wave::Wave{OneDim})::Vector{Equation}
    x, t, u = unpack(wave)
    Dx = Differential(x)
    x_min, x_max = getbounds(x)

    return [
        Dx(u(x_min, t)) ~ 0.,
        Dx(u(x_max, t)) ~ 0.]
end

"""
The neumann boundary condition which specifies that the positional derivative of the wave displacement
along the boundary of the spacial domain is always zero.
"""
function neumann(wave::Wave{TwoDim})::Vector{Equation}
    x, y, t, u = unpack(wave)
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

function time_condition(wave::Wave{OneDim})::Equation
    x, t, u = unpack(wave)
    Dt = Differential(t)
    return Dt(u(x, 0.0)) ~ 0.
end

"""
Our initial assumption that the rate of change of the wave displacement over time is zero over the
entire spacial domain of the wave.
"""
function time_condition(wave::Wave{TwoDim})::Equation
    x, y, t, u = unpack(wave)
    Dt = Differential(t)
    return Dt(u(x, y, 0.0)) ~ 0.
end

"""
The default set of boundary conditions for an acoustic wave.
"""
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

function spacetime(wave::Wave)::Vector{Num}
    return [dims(wave)..., wave.t]
end

function signature(wave::Wave)
    return wave.u(dims(wave)..., wave.t)
end

function wave_discretizer(wave::Wave, n::Int)
    return MOLFiniteDifference([Pair.(dims(wave), n)...], wave.t)
end