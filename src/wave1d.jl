export Wave1D, GaussianPulse1D, RandomGaussianPulses1D, WaveSolution1D

struct Wave1D <: AbstractWave
    x::Num
    t::Num
    u::CallWithMetadata
end

function Wave1D(;x_min, x_max, t_max)
    @parameters x [bounds = (x_min, x_max)]
    @parameters t [bounds = (0.0, t_max)]
    @variables u(..)

    return Wave1D(x, t, u)
end

function wave_equation(wave::Wave1D, C::Function)::Equation
    x, t, u = wave.x, wave.t, wave.u
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    return Dtt(u(x, t)) ~ C(x, t)^2 * Dxx(u(x, t))
end

function dirichlet(wave::Wave1D)::Vector{Equation}
    x, t, u = wave.x, wave.t, wave.u
    x_min, x_max = getbounds(x)

    return [
        u(x_min, t) ~ 0., 
        u(x_max, t) ~ 0.]
end

function neumann(wave::Wave1D)::Vector{Equation}
    x, t, u = wave.x, wave.t, wave.u
    x_min, x_max = getbounds(x)
    Dx = Differential(x)

    return [
        Dx(u(x_min, t)) ~ 0., 
        Dx(u(x_max, t)) ~ 0.]
end

function time_condition(wave::Wave1D)::Equation
    x, t, u = wave.x, wave.t, wave.u
    t_min, _ = getbounds(t)
    Dt = Differential(t)

    return Dt(u(x, t_min)) ~ 0.
end

function boundary_conditions(wave::Wave1D)
    return vcat(dirichlet(wave), neumann(wave), [time_condition(wave)])
end

struct GaussianPulse1D <: InitialCondition
    u0::Function
end

function GaussianPulse1D(intensity::Real, x_pos::Real)
    u0 = x -> exp(- intensity * (x - x_pos) ^ 2)

    return GaussianPulse1D(u0)
end

function (pulse::GaussianPulse1D)(x)
    return pulse.u0(x)
end

function Base.:+(p1::GaussianPulse1D, p2::GaussianPulse1D)::GaussianPulse1D
    u0 = x -> p1(x) + p2(x)
    return GaussianPulse1D(u0)
end

struct RandomGaussianPulses1D <: InitialCondition
    u0::GaussianPulse1D
end

function RandomGaussianPulses1D(;
        n_pulses::Int,
        intensity = nothing,
        intensity_min = nothing, intensity_max = nothing, 
        x_min = nothing, x_max = nothing)

    u0 = []

    for _ ∈ 1:n_pulses
        loc = rand(Uniform(x_min, x_max))
        
        if !isnothing(intensity_min)
            intensity = rand(Uniform(intensity_min, intensity_max))
        end

        pulse = GaussianPulse1D(intensity, loc)
        push!(u0, pulse)
    end

    return RandomGaussianPulses1D(sum(u0))
end

function (pulse::RandomGaussianPulses1D)(x)
    return pulse.u0(x)
end

struct WaveSolution1D
    x::Vector
    t::Vector
    u::Matrix
end

function animate!(sol::WaveSolution1D, path::String; ylim::Tuple = (-1.5, 1.5))

    a = Animation()

    @showprogress for i ∈ axes(sol.t, 1)

        t = sol.t[i]

        p = plot(sol.x, sol.u[:, i], ylim = ylim, label = false)
        xlabel!(p, "x")
        ylabel!(p, "u(x, $t)")
        title!(p, "Wave at Time = $t")

        frame(a, p)
    end

    gif(a, path, fps = 20)
end

function animate!(sol1::WaveSolution1D, sol2::WaveSolution1D, path::String; ylim::Tuple = (-1.5, 1.5))

    a = Animation()

    @showprogress for i ∈ axes(sol1.t, 1)

        t = sol1.t[i]

        p = plot(sol1.x, sol1.u[:, i], ylim = ylim, label = "Ground Truth")
        plot!(p, sol2.x, sol2.u[:, i], label = "Predicted")
        xlabel!(p, "x")
        ylabel!(p, "u(x, $t)")
        title!(p, "Wave at Time = $t")

        frame(a, p)
    end

    gif(a, path, fps = 20)
end