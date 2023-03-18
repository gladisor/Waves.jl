using ReinforcementLearning
using Flux
Flux.CUDA.allowscalar(false)
using IntervalSets
using CairoMakie
using Statistics: mean
using Flux.Data: DataLoader

using Waves

include("design_encoder.jl")

design_kwargs = Dict(
    :width => 1, 
    :hight => 2, 
    :spacing => 1.0f0, 
    :r => 0.5f0, 
    :c => 0.20f0, 
    :center => [0.0f0, 0.0f0])

function random_radii_scatterer_formation(;kwargs...)
    config = scatterer_formation(;kwargs)
    r = rand(Float32, size(config.r))
    r = r * (Waves.MAX_RADII - Waves.MIN_RADII) .+ Waves.MIN_RADII
    return Scatterers(config.pos, r, config.c)
end

path = "dynamic_b"

config = random_radii_scatterer_formation(;design_kwargs...)
grid_size = 4.0f0
elements = 128
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    initial_condition = PlaneWave(dim, -2.0f0, 10.0f0),
    wave = build_wave(dim, fields = fields),
    cell = WaveCell(split_wave_pml, runge_kutta),
    design = config,
    random_design = () -> random_radii_scatterer_formation(;design_kwargs...),
    space = radii_design_space(config, 0.2f0),
    design_steps = 100,
    tmax = 10.0f0;
    dim = dim,
    dynamics_kwargs...))

policy = RandomDesignPolicy(action_space(env))

@time train_data = generate_episode_data(policy, env, 50)
@time test_data = generate_episode_data(policy, env, 5)
@time val_data = generate_episode_data(policy, env, 1)

train_data_loader = DataLoader(train_data, shuffle = true)
test_data_loader = DataLoader(test_data, shuffle = true)

Flux.CUDA.device!(2)

h_fields = 128
z_fields = 2
activation = relu
z_size = Int.(size(dim) ./ (2 ^ 3))
design_size = 2 * length(vec(config))

wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation) |> gpu
design_encoder = DesignEncoder(design_size, 256, prod(z_size), activation) |> gpu

layers = Chain(
    Dense(prod(z_size), 256, activation),
    Dense(256, 256, activation),
    z -> sum(z, dims = 2),
    Dense(256, prod(z_size), sigmoid))

# cell = WaveCell(nonlinear_latent_wave, runge_kutta)
cell = WaveRNNCell(nonlinear_latent_wave, runge_kutta, layers) |> gpu
z_dim = OneDim(grid_size, prod(z_size))
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...) |> gpu
wave_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation) |> gpu

opt = Adam(0.0001)
ps = Flux.params(wave_encoder, wave_decoder, design_encoder)
train_loss = Float32[]
test_loss = Float32[]

for epoch in 1:100
    for (s, a) in train_data_loader
        s = gpu(first(s))
        a = gpu(first(a))

        u_true = cat(s.sol.total.u[2:end]..., dims = 4)

        gs = Flux.gradient(ps) do 
            z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
            latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
            z_sequence = reshape(latents, z_size..., z_fields + 1, env.design_steps)
            u_pred = wave_decoder(z_sequence)

            loss = sqrt(Flux.Losses.mse(u_true, u_pred))

            Flux.ignore() do
                push!(train_loss, loss)
                println("Train Loss: $loss")
            end

            return loss
        end

        Flux.Optimise.update!(opt, ps, gs)
    end

    batch_test_loss = []

    for (s, a) in test_data_loader
        s = gpu(first(s))
        a = gpu(first(a))

        u_true = cat(s.sol.total.u[2:end]..., dims = 4)
        
        z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
        latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
        z_sequence = reshape(latents, z_size..., z_fields + 1, env.design_steps)
        u_pred = wave_decoder(z_sequence)

        loss = sqrt(Flux.Losses.mse(u_true, u_pred))
        push!(batch_test_loss, loss)
    end

    for (i, (s, a)) in enumerate(zip(val_data...))

        s = gpu(s)
        a = gpu(a)

        u_true = cat(s.sol.total.u[2:end]..., dims = 4)
        z = hcat(wave_encoder(s.sol.total), design_encoder(s.design, a))
        latents = cat(integrate(cell, z, z_dynamics, env.design_steps)..., dims = 3)
        z_sequence = reshape(latents, z_size..., z_fields + 1, env.design_steps)
        u_pred = wave_decoder(z_sequence)

        Waves.plot_comparison!(dim, cpu(u_true), cpu(u_pred), path = joinpath(path, "comparison_$(i).png"))
    end

    push!(test_loss, mean(batch_test_loss))

    fig = Figure()
    ax1 = Axis(fig[1, 1], title = "Training Loss", xlabel = "Gradient Update")
    ax2 = Axis(fig[1, 2], title = "Testing Loss", xlabel = "Epoch")
    lines!(ax1, train_loss, color = :blue)
    scatter!(ax2, test_loss, color = :red)

    y_min = minimum(train_loss)
    y_max = maximum(train_loss)
    ylims!(ax1, y_min, y_max)
    ylims!(ax2, y_min, y_max)
    save(joinpath(path, "loss.png"), fig)
end