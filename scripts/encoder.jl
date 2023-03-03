struct WaveEncoder
    z_dynamics::WaveDynamics
    z_steps::Int
    derivative_function::Function
    integration_function::Function
    layers::Chain
end

Flux.@functor WaveEncoder
Flux.trainable(encoder::WaveEncoder) = (encoder.layers,)

function WaveEncoder(;
        z_dim::OneDim,
        derivative_function::Function,
        integration_function::Function,
        fields::Int,
        steps::Int,
        dynamics_kwargs...)

    z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...)

    elements = length(dim.x)

    layers = Chain(
        Conv((4, 4), 1 => 1, pad = SamePad(), tanh), MaxPool((4, 4)),
        Conv((4, 4), 1 => 1, pad = SamePad(), tanh), MaxPool((4, 4)),
        Flux.flatten,
        Dense(144, elements * fields, tanh),
        z -> reshape(z, elements, fields, :))

    return WaveEncoder(z_dynamics, steps, derivative_function, integration_function, layers)
end

function (encoder::WaveEncoder)(u::AbstractMatrix{Float32})
    return u
end

# encoder = WaveEncoder(
#     z_dim = OneDim(5.0f0, 100),
#     derivative_function = latent_wave,
#     integration_function = runge_kutta,
#     fields = 2,
#     steps = 1;
#     dynamics_kwargs...)
