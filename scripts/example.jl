using Waves
using CairoMakie: save
using Flux

function build_wave(dim::AbstractDim; fields::Int)
    return zeros(Float32, size(dim)..., fields)
end

function latent_wave(wave::AbstractMatrix{Float32}, t::Float32, dynamics::WaveDynamics)
    U = wave[:, 1]
    V = wave[:, 2]
    b = wave[:, 3]

    ∇ = dynamics.grad
    σx = dynamics.pml

    dU = b .* (∇ * V) .- σx .* U
    dV = ∇ * U .- σx .* V

    return cat(dU, dV, b, dims = 2)
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

## Set some parameters
grid_size = 5.0f0
elements = 300
steps = 200

## Create the dimensional space
dim = OneDim(grid_size, elements)

## Make initial condition
pulse = Pulse(dim, 0.0f0, 10.0f0) |> gpu

## Initialize wave
wave = build_wave(dim, fields = 3) |> gpu
wave = pulse(wave)

## Split wave integration scheme with runge_kutta integrator
activation = tanh

layers =  Chain(
    Dense(length(dim.x), length(dim.x), activation),
    b -> sum(b, dims = 2),
    sigmoid
    )

cell = WaveRNNCell(latent_wave, runge_kutta, layers) |> gpu
dynamics = WaveDynamics(dim = dim, pml_width = 1.0f0, pml_scale = 50.0f0, ambient_speed = 1.0f0, dt = 0.01f0) |> gpu

opt = Adam(0.001)

sol = solve(cell, wave, dynamics, steps) |> cpu
render!(sol, path = "vid.mp4")

wave = pulse(wave)

ps = Flux.params(cell)

for i ∈ 1:100

    Waves.reset!(dynamics)
    gs = Flux.gradient(ps) do 
        latents = integrate(cell, wave, dynamics, steps)
        e = sum(energy(displacement(latents[end])))

        Flux.ignore() do 
            println(e)
        end

        return e
    end

    Flux.Optimise.update!(opt, ps, gs)
end

wave = pulse(wave)

sol = solve(cell, wave, dynamics, steps)
render!(sol, path = "vid_opt.mp4")