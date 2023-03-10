using Flux
using Flux.Losses: mse
Flux.CUDA.allowscalar(false)
using CairoMakie
using Statistics: mean
using Distributions: Uniform

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

grid_size = 5.0f0
elements = 200
fields = 6

pulse_x = 0.0f0
pulse_y = 2.0f0
pulse_intensity = 5.0f0

h_fields = 32
activation = tanh
z_fields = 3
steps = 100

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)
dim = TwoDim(grid_size, elements)
latent_dim = OneDim(grid_size, elements)

layers =  Chain(
    Dense(length(dim.x), h_fields, activation),
    Dense(h_fields, length(dim.x), activation),
    b -> sum(b, dims = 2),
    Dense(length(dim.x), length(dim.x), sigmoid))

encoder = WaveEncoder(
    wave_fields = fields, 
    h_fields = h_fields, 
    latent_fields = z_fields,
    wave_dim = dim, 
    latent_dim = latent_dim, 
    activation = activation, 
    cell = WaveRNNCell(latent_wave, runge_kutta, layers) |> gpu,
    # cell = cell,
    dynamics = WaveDynamics(dim = latent_dim; dynamics_kwargs...) |> gpu
    ) |> gpu

decoder = WaveCNNDecoder(
    wave_fields = fields, 
    h_fields = h_fields, 
    latent_fields = z_fields,
    wave_dim = dim, 
    latent_dim = latent_dim, 
    activation = activation) |> gpu

dynamics = WaveDynamics(dim = dim; dynamics_kwargs...) |> gpu

wave = zeros(Float32, size(dim)..., fields)

x = WaveSol[]
y = []

pulse_x = Uniform(-2.0f0, 2.0f0)

for i ∈ 1:100
    Waves.reset!(dynamics)
    pulse = Pulse(dim, Float32(rand(pulse_x)), Float32(rand(pulse_x)), pulse_intensity)
    sol = solve(cell, gpu(pulse(wave)), dynamics, steps) |> cpu
    u_true = cat(sol.u[2:end]..., dims = 4)
    push!(x, sol)
    push!(y, u_true)
end

opt = Adam(0.0005)
ps = Flux.params(encoder, decoder)

train_loss = Float32[]

for i ∈ 1:10

    for i in axes(x, 1)
        Waves.reset!(encoder.dynamics)

        gs = Flux.gradient(ps) do 
            u_pred = decoder(encoder(gpu(x[i]), steps))
            loss = sqrt(mse(gpu(y[i]), u_pred))
    
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
    u_pred = decoder(encoder(gpu(x[i]), steps))
    plot_comparison!(cpu(y[i]), cpu(u_pred), path = "non_linear_rmse_comparison_$i.png")
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, train_loss, linewidth = 3)
save("non_linear_loss_tanh.png", fig)