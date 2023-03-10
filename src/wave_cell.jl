export integrate, solve, WaveCell, WaveRNNCell

function integrate(cell::AbstractWaveCell, wave::AbstractArray{Float32}, dynamics::WaveDynamics, steps::Int)
    iter = Flux.Recur(cell, wave)
    return [iter(dynamics) for _ ∈ 1:steps]
end

function solve(cell::AbstractWaveCell, wave::AbstractArray{Float32}, dynamics::WaveDynamics, steps::Int)
    u = integrate(cell, wave, dynamics, steps)
    pushfirst!(u, wave)
    t = collect(range(0.0f0, dynamics.dt * steps, steps + 1))
    return WaveSol(dynamics.dim, t, u)
end

struct WaveCell <: AbstractWaveCell
    derivative_function::Function
    integration_function::Function
end

Flux.@functor WaveCell

function (cell::WaveCell)(z::AbstractArray{Float32}, dynamics::WaveDynamics)
    z′ = z .+ cell.integration_function(cell.derivative_function, z, dynamics)
    dynamics.t += 1
    return z′, z′
end

struct WaveRNNCell <: AbstractWaveCell
    derivative_function::Function
    integration_function::Function
    layers::Chain
end

Flux.@functor WaveRNNCell
Flux.trainable(cell::WaveRNNCell) = (cell.layers,)

function (cell::WaveRNNCell)(z::AbstractMatrix{Float32}, dynamics::WaveDynamics)
    b = cell.layers(z)
    z = cat(z[:, [1, 2]], b, dims = 2)
    z = z .+ cell.integration_function(cell.derivative_function, z, dynamics)
    dynamics.t += 1
    return z, z
end