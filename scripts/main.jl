using Flux
using Flux.Losses: mse
using ReinforcementLearning
using CairoMakie

using Waves
using Waves: random_radii_scatterer_formation

function get_target_u(sol::WaveSol)
    return cat(sol.u[2:end]..., dims = ndims(first(sol.u)) + 1)
end

include("design_encoder.jl")
include("wave_net.jl")
include("inc_sc_wave_net.jl")

function Waves.plot_comparison!(
        u_inc_pred::AbstractArray, u_sc_pred::AbstractArray, 
        u_inc_true::AbstractArray, u_sc_true::AbstractArray, 
        idx::Int; path::String)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Incident Wave", ylabel = "Y (m)", xticklabelsvisible = false, xticksvisible = false)
    ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Incident Wave", yticklabelsvisible = false, yticksvisible = false, xticklabelsvisible = false, xticksvisible = false)
    ax3 = Axis(fig[2, 1], aspect = 1.0, title = "True Scattered Wave", xlabel = "X (m)", ylabel = "Y (m)")
    ax4 = Axis(fig[2, 2], aspect = 1.0, title = "Predicted Scattered Wave", xlabel = "X (m)", yticklabelsvisible = false, yticksvisible = false)

    heatmap!(ax1, dim.x, dim.y, u_inc_true[:, :, 1, idx], colormap = :ice)
    heatmap!(ax2, dim.x, dim.y, u_inc_pred[:, :, 1, idx], colormap = :ice)
    heatmap!(ax3, dim.x, dim.y, u_sc_true[:, :, 1, idx], colormap = :ice)
    heatmap!(ax4, dim.x, dim.y, u_sc_pred[:, :, 1, idx], colormap = :ice)

    save(path, fig)
    return nothing
end

function Waves.plot_comparison!(model::IncScWaveNet, s::WaveEnvState, a::AbstractDesign; path::String)
    u_inc_pred, u_sc_pred = cpu(model(s, a))
    u_inc_true = get_target_u(s.sol.incident) |> cpu
    u_sc_true = get_target_u(s.sol.scattered) |> cpu
    plot_comparison!(u_inc_pred, u_sc_pred, u_inc_true, u_sc_true, length(s.sol.total), path = path)
end

function plot_loss!(train_loss::Vector{Float32}; path::String)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Gradient Update", ylabel = "Loss", title = "Training Loss", aspect = 1.0)
    lines!(ax, train_loss, linewidth = 3)
    save(path, fig)
end

function plot_loss!(train_loss::Vector{Float32}, test_loss::Vector{Float32}; path::String)
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel = "Gradient Update", ylabel = "Loss", title = "Training Loss", aspect = 1.0)
    ax2 = Axis(fig[1, 2], xlabel = "Epoch", ylabel = "Average Loss", title = "Average Testing Loss per Epoch", aspect = 1.0)
    lines!(ax1, train_loss, linewidth = 3, color = :blue)
    lines!(ax2, test_loss, linewidth = 3, color = :orange)
    save(path, fig)
end

design_kwargs = Dict(:width => 1, :hight => 1, :spacing => 1.0f0, :c => 0.20f0, :center => [0.0f0, 0.0f0])
config = random_radii_scatterer_formation(;design_kwargs...)
grid_size = 4.0f0
elements = 128
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    initial_condition = PlaneWave(dim, -2.0f0, 10.0f0),
    # initial_condition = Pulse(dim, -3.0f0, 0.0f0, 10.0f0),
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
@time data = generate_episode_data(policy, env, 1)
train_loader = Flux.DataLoader(data, shuffle = true)

model = gpu(IncScWaveNet(
        fields = fields, 
        h_fields = 128, 
        z_fields = 2, 
        activation = relu,
        design_size = 2 * length(vec(config)), 
        h_size = 256, 
        grid_size = grid_size,
        z_elements = prod(Int.(size(dim) ./ (2 ^ 3))); 
        dynamics_kwargs...))

opt = Adam(0.0001)
ps = Flux.params(model)
train_loss_history = Float32[]

for epoch in 1:2

    for (s, a) in train_loader

        s, a = gpu(s[1]), gpu(a[1]) ## batch size is one

        gs = Flux.gradient(ps) do
            train_loss = loss(model, s, a)

            Flux.ignore() do
                push!(train_loss_history, train_loss)
                println("Epoch: $epoch, Train Loss: $train_loss")
            end

            return train_loss
        end

        Flux.Optimise.update!(opt, ps, gs)
    end

    s_sample = gpu(data[1][1])
    a_sample = gpu(data[2][1])

    plot_comparison!(model, s_sample, a_sample, path = "comparison.png")
    plot_loss!(train_loss_history, path = "train_loss.png")
end