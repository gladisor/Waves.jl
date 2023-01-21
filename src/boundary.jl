export OpenBoundary, ClosedBoundary, PlaneWave

struct OpenBoundary <: WaveBoundary end

function (boundary::OpenBoundary)(wave::Wave)
    return open_boundary(wave)
end

struct ClosedBoundary <: WaveBoundary end

function (boundary::ClosedBoundary)(wave::Wave)
    return closed_boundary(wave)
end

struct PlaneWave <: WaveBoundary 
    intensity::Real
    shift::Real
end

function PlaneWave()
    return PlaneWave(5.0, 1.1)
end

function (boundary::PlaneWave)(wave::Wave)
    x_min, _ = getbounds(wave.dim.x)
    bcs = open_boundary(wave)
    bcs[1] = wave.u(x_min, spacetime(wave)[2:end]...) ~ exp(-boundary.intensity*(wave.t-boundary.shift)^2)
    return bcs
end