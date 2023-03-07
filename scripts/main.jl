using Flux
using CairoMakie

using Waves

grid_size = 5.0f0
elements = 200
fields = 6

pulse_x = 0.0f0
pulse_y = 2.0f0
pulse_intensity = 5.0f0

h_size = 1
activation = relu
z_elements = 200
z_fields = 2

steps = 100

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)

dim = TwoDim(grid_size, elements)
dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)

pulse = Pulse(dim, pulse_x, pulse_y, pulse_intensity)
wave = zeros(Float32, size(dim)..., fields)
wave = pulse(wave)

encoder_layers = Chain(
    Conv((3, 3), 6 => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    MaxPool((2, 2)),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    MaxPool((2, 2)),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(625, z_elements * z_fields, activation),
    z -> reshape(z, z_elements, z_fields)
)

decoder_layers = Chain(
    Dense(400, 625),
    z -> reshape(z, 25, 25, 1, :),
    Conv((3, 3), 1 => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Upsample((2, 2)),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Upsample((2, 2)),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Conv((3, 3), h_size => h_size, activation, pad = SamePad()),
    Upsample((2, 2)),
    Conv((3, 3), h_size => fields, activation, pad = SamePad())) |> gpu

cell = WaveCell(split_wave_pml, runge_kutta)
z_dim = OneDim(grid_size, z_elements)
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...)
encoder = WaveEncoder(cell, z_dynamics, encoder_layers) |> gpu

Waves.reset!(dynamics)
u = integrate(cell, wave, dynamics, steps)
u_true = cat(u..., dims = 4)

pushfirst!(u, wave)
t = collect(range(0.0f0, dynamics.dt * steps, steps + 1))
sol = WaveSol(dim, t, u)

opt = Adam(0.001)
ps = Flux.params(encoder, decoder_layers)

for i ∈ 1:100
    Waves.reset!(encoder.dynamics)

    gs = Flux.gradient(ps) do 
        u_z = encoder(sol, steps)
        u_pred = decoder_layers(u_z)
        loss = sqrt.(mean((y_true .- y_pred) .^ 2))

        Flux.ignore() do 
            println("Loss: $loss")
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end