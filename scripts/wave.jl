
struct Wave{D <: AbstractDim}
    u::AbstractArray{Float32}
end

Flux.@functor Wave

function Wave(dim::D, fields::Int = 1) where D <: AbstractDim
    u = zeros(Float32, size(dim)..., fields)
    return Wave{D}(u)
end

function Base.size(wave::Wave)
    return size(wave.u)
end

function Base.getindex(wave::Wave, idx...)
    return wave.u[idx...]
end

function Base.axes(wave::Wave, idx)
    return axes(wave.u, idx)
end

function energy(wave::Wave{OneDim})
    return sum(wave[:, 1] .^ 2)
end

function Waves.Pulse(dim::OneDim; x::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    return Pulse(dim, x, intensity)
end

function Waves.Pulse(dim::TwoDim; x::Float32 = 0.0f0, y::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    return Pulse(dim, x, y, intensity)
end

function (pulse::Pulse{OneDim})(wave::Wave{OneDim})
    u0 = exp.(-pulse.intensity * pulse.mesh_grid .^ 2)
    return Wave{OneDim}(hcat(u0, wave[:, 2:end]))
end

function (pulse::Pulse{TwoDim})(wave::Wave{TwoDim})
    u0 = dropdims(exp.(-pulse.intensity * sum(pulse.mesh_grid .^ 2, dims = 3)), dims = 3)
    return Wave{TwoDim}(cat(u0, wave[:, :, 2:end], dims = 3))
end

function Waves.split_wave_pml(wave::Wave{OneDim}, t::Float32, dynamics::WaveDynamics)
    u = wave[:, 1]
    v = wave[:, 2]
    ∇ = dynamics.grad

    ∇ = dynamics.grad
    σx = dynamics.pml
    C = Waves.speed(dynamics, t)
    b = C .^ 2

    du = b .* (∇ * v) .- σx .* u
    dvx = ∇ * u .- σx .* v

    return Wave{OneDim}(cat(du, dvx, dims = 2))
end