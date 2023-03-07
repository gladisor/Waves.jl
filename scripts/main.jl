using Flux
using CairoMakie
using Statistics: mean

using Waves

include("../src/models/wave_encoder.jl")
include("../src/models/wave_cnn_decoder.jl")

grid_size = 5.0f0
elements = 200
fields = 6

pulse_x = 0.0f0
pulse_y = 2.0f0
pulse_intensity = 5.0f0

h_size = 12
activation = relu
z_elements = 200
z_fields = 2

steps = 100

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)

dim = TwoDim(grid_size, elements)
dynamics = WaveDynamics(dim = dim; dynamics_kwargs...) |> gpu

pulse = Pulse(dim, pulse_x, pulse_y, pulse_intensity)
wave = zeros(Float32, size(dim)..., fields)
wave = pulse(wave) |> gpu

cell = WaveCell(split_wave_pml, runge_kutta)
z_dim = OneDim(grid_size, z_elements)
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...) |> gpu

encoder = WaveEncoder(
    wave_fields = 6,
    h_fields = 12,
    latent_fields = 2,
    wave_dim = dim,
    latent_dim = z_dim,
    activation = relu,
    cell = cell,
    dynamics = z_dynamics
)

decoder = WaveCNNDecoder(
    wave_fields = 6,
    h_fields = 12,
    latent_fields = 2,
    wave_dim = dim,
    latent_dim = z_dim,
    activation = relu
)

Waves.reset!(dynamics)
u = integrate(cell, wave, dynamics, steps)
u_true = cat(u..., dims = 4)
pushfirst!(u, wave)
t = collect(range(0.0f0, dynamics.dt * steps, steps + 1))
sol = WaveSol(dim, t, u)

opt = Adam(0.0005)
ps = Flux.params(encoder, decoder)

for i ∈ 1:500
    Waves.reset!(encoder.dynamics)

    gs = Flux.gradient(ps) do 
        u_pred = decoder_layers(encoder(sol, steps))
        loss = sqrt(Flux.Losses.mse(u_true, u_pred))

        Flux.ignore() do 
            println("Loss: $loss")
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end

function plot_comparison!(y_true, y_pred; path::String)
    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = AxisAspect(1.0))
    heatmap!(ax1, dim.x, dim.y, y_true[:, :, 1, end], colormap = :ice)
    ax2 = Axis(fig[1, 2], aspect = AxisAspect(1.0))
    heatmap!(ax2, dim.x, dim.y, y_pred[:, :, 1, end], colormap = :ice)
    ax3 = Axis(fig[2, 1], aspect = AxisAspect(1.0))
    heatmap!(ax3, dim.x, dim.y, y_true[:, :, 1, end ÷ 2], colormap = :ice)
    ax4 = Axis(fig[2, 2], aspect = AxisAspect(1.0))
    heatmap!(ax4, dim.x, dim.y, y_pred[:, :, 1, end ÷ 2], colormap = :ice)
    ax5 = Axis(fig[3, 1], aspect = AxisAspect(1.0))
    heatmap!(ax5, dim.x, dim.y, y_true[:, :, 1, 1], colormap = :ice)
    ax6 = Axis(fig[3, 2], aspect = AxisAspect(1.0))
    heatmap!(ax6, dim.x, dim.y, y_pred[:, :, 1, 1], colormap = :ice)
    save(path, fig)
end

u_pred = decoder(encoder(sol, steps))
plot_comparison!(cpu(u_true), cpu(u_pred), path = "rmse_comparison.png")

