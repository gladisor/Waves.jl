export conditions, GaussianPulse

function conditions(wave::Wave, ic::InitialCondition)
    return [
        wave.u(dims(wave)..., 0.0) ~ ic(wave),
        boundary_conditions(wave)...]
end

struct GaussianPulse <: InitialCondition
    intensity::Real
end

function (pulse::GaussianPulse)(wave::Wave{OneDim})
    return exp(-pulse.intensity * (wave.dim.x ^ 2))
end

function (pulse::GaussianPulse)(wave::Wave{TwoDim})
    return exp(-pulse.intensity * (wave.dim.x ^ 2 + wave.dim.y ^ 2))
end

function (pulse::GaussianPulse)(wave::Wave{ThreeDim})
    return exp(-pulse.intensity * (wave.dim.x ^ 2 + wave.dim.y ^ 2 + wave.dim.z ^ 2))
end