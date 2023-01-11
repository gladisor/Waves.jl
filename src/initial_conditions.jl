export conditions, GaussianPulse

function conditions(wave::Wave, ic::InitialCondition)
    return [
        wave.u(dims(wave)..., 0.0) ~ ic(wave),
        boundary_conditions(wave)...]
end

struct GaussianPulse <: InitialCondition
    intensity::Real
    loc::Vector
end

function GaussianPulse(;intensity, loc = zeros(3))
    return GaussianPulse(intensity, loc)
end

function (pulse::GaussianPulse)(wave::Wave{OneDim})
    return exp(-pulse.intensity * ((wave.dim.x - pulse.loc[1]) ^ 2))
end

function (pulse::GaussianPulse)(wave::Wave{TwoDim})
    return exp(-pulse.intensity * ((wave.dim.x - pulse.loc[1]) ^ 2 + (wave.dim.y - pulse.loc[2]) ^ 2))
end

function (pulse::GaussianPulse)(wave::Wave{ThreeDim})
    return exp(-pulse.intensity * ((wave.dim.x - pulse.loc[1]) ^ 2 + (wave.dim.y - pulse.loc[2]) ^ 2 + (wave.dim.z - pulse.loc[3]) ^ 2))
end