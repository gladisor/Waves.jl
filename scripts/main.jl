using Flux
using Flux.Losses: mse
Flux.CUDA.allowscalar(false)
using CairoMakie
using Statistics: mean

using Waves

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

function solve(cell::WaveCell, wave::AbstractArray{Float32}, dynamics::WaveDynamics, steps::Int)
    u = integrate(cell, wave, dynamics, steps)
    pushfirst!(u, wave)
    t = collect(range(0.0f0, dynamics.dt * steps, steps + 1))
    return WaveSol(dim, t, u)
end

# function solve(model, wave::AbstractArray{Float32}, dynamics::WaveDynamics, steps::Int)

# end

include("../src/models/wave_encoder.jl")
include("../src/models/wave_cnn_decoder.jl")

grid_size = 5.0f0
elements = 200
fields = 6

pulse_x = 0.0f0
pulse_y = 2.0f0
pulse_intensity = 5.0f0

h_fields = 12
activation = tanh
z_fields = 2
steps = 100

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)

dim = TwoDim(grid_size, elements)
cell = WaveCell(split_wave_pml, runge_kutta)
latent_dim = OneDim(grid_size, elements)

encoder = WaveEncoder(
    wave_fields = fields,
    h_fields = h_fields,
    latent_fields = z_fields,
    wave_dim = dim,
    latent_dim = latent_dim,
    activation = activation,
    cell = cell,
    dynamics = WaveDynamics(dim = latent_dim; dynamics_kwargs...) |> gpu
) |> gpu

decoder = WaveCNNDecoder(
    wave_fields = fields,
    h_fields = h_fields,
    latent_fields = z_fields,
    wave_dim = dim,
    latent_dim = latent_dim,
    activation = activation
) |> gpu

dynamics = WaveDynamics(dim = dim; dynamics_kwargs...) |> gpu
pulse = Pulse(dim, pulse_x, pulse_y, pulse_intensity) |> gpu

wave = zeros(Float32, size(dim)..., fields) |> gpu
wave = pulse(wave)

Waves.reset!(dynamics)
sol = solve(cell, wave, dynamics, steps)
u_true = cat(sol.u[2:end]..., dims = 4)

Waves.reset!(dynamics)
pulse2 = Pulse(dim, -2.0f0, 2.0f0, pulse_intensity) |> gpu
sol2 = solve(cell, pulse2(wave), dynamics, steps)
u_true2 = cat(sol2.u[2:end]..., dims = 4)

Waves.reset!(dynamics)
pulse3 = Pulse(dim, 2.0f0, -2.0f0, pulse_intensity) |> gpu
sol3 = solve(cell, pulse3(wave), dynamics, steps)
u_true3 = cat(sol3.u[2:end]..., dims = 4)

x = [sol, sol2, sol3]
y = [u_true, u_true2, u_true3]

opt = Adam(0.0005)
ps = Flux.params(encoder, decoder)

train_loss = Float32[]

for i ∈ 1:100

    for i in axes(x, 1)
        Waves.reset!(encoder.dynamics)

        gs = Flux.gradient(ps) do 
            u_pred = decoder(encoder(x[i], steps))
            loss = sqrt(mse(y[i], u_pred))
    
            Flux.ignore() do 
                println("Loss: $loss")
                push!(train_loss, loss)
            end
    
            return loss
        end
    
        Flux.Optimise.update!(opt, ps, gs)
    end
end

for i in axes(x, 1)
    u_pred = decoder(encoder(x[i], steps))
    plot_comparison!(cpu(y[i]), cpu(u_pred), path = "rmse_comparison_$i.png")
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, train_loss, linewidth = 3)
save("loss_tanh.png", fig)