using Flux
using Flux.Losses: mse
Flux.CUDA.allowscalar(false)
using CairoMakie
using Statistics: mean
using Distributions: Uniform

using Waves
using Waves: InitialCondition

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
z_fields = 2
steps = 300

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 50.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)
dim = TwoDim(grid_size, elements)
latent_dim = OneDim(grid_size, elements)

layers =  Chain(
    Dense(elements, h_fields, activation),
    Dense(h_fields, elements, activation),
    b -> sum(b, dims = 2),
    Dense(elements, elements, sigmoid))

model_kwargs = Dict(
    :wave_fields => fields,
    :h_fields => h_fields,
    :latent_fields => z_fields,
    :wave_dim => dim,
    :latent_dim => latent_dim,
    :activation => activation)

encoder = WaveEncoder(
    wave_fields = fields, 
    h_fields = h_fields, 
    latent_fields = z_fields,
    wave_dim = dim, 
    latent_dim = latent_dim, 
    activation = activation, 
    # cell = WaveRNNCell(latent_wave, runge_kutta, layers) |> gpu,
    cell = cell,
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

wave = build_wave(dim, fields = fields)

wave = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)(wave)

p = WavePlot(dim)
plot_wave!(p, dim, wave)
save("u.png", p.fig)

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

    random_pulse.pulse = Pulse(
        random_pulse.pulse.dim,
        Float32(rand(random_pulse.x_distribution)), 
        Float32(rand(random_pulse.y_distribution)),
        random_pulse.pulse.intensity
        )
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

random_pulse = RandomPulseTwoDim(
    dim,
    Uniform(-4.0f0, 4.0f0), 
    Uniform(-4.0f0, 4.0f0), 
    10.0f0
    ) |> gpu

wave = build_wave(dim, fields = 6) |> gpu
wave = random_pulse(wave)

@time sol = solve(cell, wave, dynamics, steps) |> cpu
@time render!(sol, path = "vid.mp4")

# function build_random_pulse_dataset(x_range::Tuple, y_range::Tuple, intensity::Float32, dynamics::WaveDynamics, steps::Int, data_size::Int)
#     wave = build_wave(dim, 6)
#     pulse_x = Uniform(x_range...)
#     pulse_y = Uniform(y_range...)

#     solutions = WaveSol[]

#     for _ ∈ 1:data_size
#         wave = Pulse(dynamics.dim, rand(pulse_x), rand(pulse_y), intensity)(wave)
#         sol = solve(cell, gpu(wave), dynamics, steps) |> cpu
#         push!(solutions, sol)
#     end

#     return solutions
# end





# opt = Adam(0.0005)
# ps = Flux.params(encoder, decoder)

# train_loss = Float32[]
# path = "long_results_linear"

# for i ∈ 1:10

#     for i in axes(x, 1)
#         Waves.reset!(encoder.dynamics)

#         gs = Flux.gradient(ps) do 
#             u_pred = decoder(encoder(gpu(x[i]), steps))
#             loss = sqrt(mse(gpu(y[i]), u_pred))
    
#             Flux.ignore() do 
#                 println("Loss: $loss")
#                 push!(train_loss, loss)
#             end
    
#             return loss
#         end
    
#         Flux.Optimise.update!(opt, ps, gs)
#     end

#     for i in axes(x, 1)
#         if i == 100
#             break
#         end
#         u_pred = decoder(encoder(gpu(x[i]), steps))
#         plot_comparison!(cpu(y[i]), cpu(u_pred), path = joinpath(path, "non_linear_rmse_comparison_$i.png"))
#     end

#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     lines!(ax, train_loss, linewidth = 3)
#     save(joinpath(path, "non_linear_loss_tanh.png"), fig)
# end



