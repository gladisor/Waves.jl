export pulse, Pulse, RandomPulseTwoDim, PlaneWave

function reset!(ic::InitialCondition)
    return nothing
end

struct Pulse{D <: AbstractDim} <: InitialCondition
    mesh_grid::AbstractArray{Float32}
    pos::AbstractVector{Float32}
    intensity::Float32
end

function Pulse(dim::OneDim, x::Float32, intensity::Float32)
    return Pulse{OneDim}(build_grid(dim), [x], intensity)
end

function Pulse(dim::TwoDim, x::Float32, y::Float32, intensity::Float32)
    return Pulse{TwoDim}(build_grid(dim), [x, y], intensity)
end

function (pulse::Pulse{OneDim})()
    return exp.(- pulse.intensity * (pulse.mesh_grid .- pulse.pos) .^ 2)
end

function (pulse::Pulse{OneDim})(wave::AbstractMatrix{Float32})
    u = pulse()
    z = pulse.mesh_grid * 0.0f0
    z = repeat(z, 1, size(wave, 2) - 1)
    return cat(u, z, dims = 2)
end

function (pulse::Pulse{TwoDim})()
    pos = reshape(pulse.pos, 1, 1, size(pulse.pos)...)
    u = exp.(- pulse.intensity * dropdims(sum((pulse.mesh_grid .- pos) .^ 2, dims = 3), dims = 3))
    return u
end

function (pulse::Pulse{TwoDim})(wave::AbstractArray{Float32, 3})
    u = pulse()
    z = dropdims(sum(pulse.mesh_grid, dims = 3), dims = 3) * 0.0f0
    z = repeat(z, 1, 1, size(wave, 3) - 1)
    return cat(u, z, dims = 3)
end

function Flux.gpu(pulse::Pulse{D}) where D <: AbstractDim
    return Pulse{D}(gpu(pulse.mesh_grid), gpu(pulse.pos), pulse.intensity)
end

function Flux.cpu(pulse::Pulse{D}) where D <: AbstractDim
    return Pulse{D}(cpu(pulse.mesh_grid), cpu(pulse.pos), pulse.intensity)
end

mutable struct RandomPulseTwoDim <: InitialCondition
    x_distribution::Uniform
    y_distribution::Uniform
    pulse::Pulse
end

function RandomPulseTwoDim(
        dim::TwoDim,
        x_distribution::Uniform,
        y_distribution::Uniform,
        intensity::Float32)
    
    pulse = Pulse(
        dim, 
        Float32(rand(x_distribution)), 
        Float32(rand(y_distribution)), 
        intensity)

    return RandomPulseTwoDim(x_distribution, y_distribution, pulse)
end

function Waves.reset!(random_pulse::RandomPulseTwoDim)
    random_pulse.pulse = Pulse{TwoDim}(
        random_pulse.pulse.mesh_grid,
        [Float32(rand(random_pulse.x_distribution)), Float32(rand(random_pulse.y_distribution))],
        random_pulse.pulse.intensity)
    return nothing
end

function (random_pulse::RandomPulseTwoDim)(wave::AbstractArray{Float32, 3})
    return random_pulse.pulse(wave)
end

function Flux.gpu(random_pulse::RandomPulseTwoDim)
    random_pulse.pulse = gpu(random_pulse.pulse)
    return random_pulse
end

function Flux.cpu(random_pulse::RandomPulseTwoDim)
    random_pulse.pulse = cpu(random_pulse.pulse)
    return random_pulse
end


struct PlaneWave <: InitialCondition
    mesh_grid::AbstractArray{Float32}
    x::Float32
    intensity::Float32
end

function PlaneWave(dim::TwoDim, x::Float32, intensity::Float32)
    return PlaneWave(build_grid(dim), x, intensity)
end

function (ic::PlaneWave)()
    return exp.(- ic.intensity * (ic.mesh_grid[:, :, 1] .- ic.x) .^ 2)
end

function (ic::PlaneWave)(wave::AbstractArray{Float32, 3})
    u = ic()
    z = dropdims(sum(ic.mesh_grid, dims = 3), dims = 3) * 0.0f0
    z = repeat(z, 1, 1, size(wave, 3) - 1)
    return cat(u, z, dims = 3)
end

function Flux.gpu(ic::PlaneWave)
    return PlaneWave(gpu(ic.mesh_grid), ic.x, ic.intensity)
end

function Flux.cpu(ic::PlaneWave)
    return PlaneWave(cpu(ic.mesh_grid), ic.x, ic.intensity)
end