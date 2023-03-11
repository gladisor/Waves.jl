using Flux
using Flux.Losses: mse
Flux.CUDA.allowscalar(false)
using CairoMakie
using Statistics: mean
using Distributions: Uniform

using Waves
using Waves: InitialCondition, AbstractWaveCell

include("../src/env.jl")

struct TotalWaveSol
    incident::WaveSol
    total::WaveSol
    scattered::WaveSol
end

function generate_solutions(
        initial_condition::InitialCondition, 
        cell::AbstractWaveCell,
        wave::AbstractArray{Float32}, 
        dynamics::WaveDynamics, 
        steps::Int,
        num_solutions::Int)

    solutions = FullWaveSol[]

    for _ ∈ 1:num_solutions
        Waves.reset!(initial_condition)
        Waves.reset!(dynamics)

        sol = solve(cell, initial_condition(wave), dynamics, steps) |> cpu
        push!(solutions, sol)
    end

    return solutions
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
steps = 200

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 50.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)
dim = TwoDim(grid_size, elements)
latent_dim = OneDim(grid_size, elements)

# layers =  Chain(
#     Dense(elements, h_fields, activation),
#     Dense(h_fields, elements, activation),
#     b -> sum(b, dims = 2),
#     Dense(elements, elements, sigmoid))

# encoder = WaveEncoder(
#     wave_fields = fields, 
#     h_fields = h_fields, 
#     latent_fields = z_fields,
#     wave_dim = dim, 
#     latent_dim = latent_dim, 
#     activation = activation, 
#     # cell = WaveRNNCell(latent_wave, runge_kutta, layers) |> gpu,
#     cell = cell,
#     dynamics = WaveDynamics(dim = latent_dim; dynamics_kwargs...) |> gpu
#     ) |> gpu

# decoder = WaveCNNDecoder(
#     wave_fields = fields, 
#     h_fields = h_fields, 
#     latent_fields = z_fields,
#     wave_dim = dim, 
#     latent_dim = latent_dim, 
#     activation = activation) |> gpu

cyl = Cylinder([0.0f0, 0.0f0], 1.0f0, 0.1f0)
dynamics = WaveDynamics(dim = dim, design = cyl; dynamics_kwargs...) |> gpu

wave = build_wave(dim, fields = fields)

# wave = Pulse(dim, 0.0f0, 0.0f0, 10.0f0)(wave)

random_pulse = RandomPulseTwoDim(
    dim,
    Uniform(-4.0f0, 4.0f0), 
    Uniform(-4.0f0, 4.0f0), 
    10.0f0
    )

wave = build_wave(dim, fields = 6) |> gpu

solutions = generate_solutions(random_pulse, cell, wave, dynamics, steps, 1)
render!(solutions[end], path = "vid.mp4")

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



