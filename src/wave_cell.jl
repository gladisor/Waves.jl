export WaveCell

struct WaveCell
    derivative_function::Function
    integration_function::Function
end

Flux.@functor WaveCell

function (cell::WaveCell)(z::AbstractArray{Float32}, dynamics::WaveDynamics)
    z′ = z .+ cell.integration_function(cell.derivative_function, z, dynamics)
    dynamics.t += 1
    return z′, z′
end

function integrate(cell::WaveCell, wave::AbstractArray{Float32}, dynamics::WaveDynamics, steps::Int)
    iter = Flux.Recur(cell, wave)
    return [iter(dynamics) for _ ∈ 1:steps]
end