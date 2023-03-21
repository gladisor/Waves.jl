using Flux
using Flux.Losses: mse
using ReinforcementLearning
using CairoMakie

using Waves
using Waves: random_radii_scatterer_formation

include("design_encoder.jl")

struct IncScWaveNet
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder

    z_cell::WaveCell
    z_dynamics::AbstractDynamics

    inc_decoder::WaveDecoder
    sc_decoder::WaveDecoder
end

Flux.@functor IncScWaveNet (wave_encoder, design_encoder, inc_decoder, sc_decoder)

function IncScWaveNet(;fields::Int, h_fields::Int, z_fields::Int, activation::Function, design_size::Int, h_size::Int, grid_size::Float32, z_elements::Int, dynamics_kwargs...)

    wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
    design_encoder = DesignEncoder(design_size, h_size, z_elements, activation)

    z_cell = WaveCell(nonlinear_latent_wave, runge_kutta)
    z_dynamics = WaveDynamics(dim = OneDim(grid_size, z_elements); dynamics_kwargs...)

    inc_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation)
    sc_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation)

    return IncScWaveNet(wave_encoder, design_encoder, z_cell, z_dynamics, inc_decoder, sc_decoder)
end

function (model::IncScWaveNet)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign, steps::Int)
    z = model.wave_encoder(wave)
    b = model.design_encoder(design, action)
    latents = integrate(model.z_cell, cat(z, b, dims = 2), model.z_dynamics, steps)
    z_concat = cat(latents..., dims = 3)
    n = Int(sqrt(size(z_concat, 1)))
    z_feature_maps = reshape(z_concat, n, n, size(z_concat, 2), :)
    incident = model.inc_decoder(z_feature_maps)
    scattered = model.sc_decoder(z_feature_maps)
    return (incident, scattered)
end

"""
Used for training when we have access to a state with an initial condition and the action
applied over a number of time steps. Takes the first waveform and decodes a prediction of the
entire trajectory.
"""
function (model::IncScWaveNet)(s::WaveEnvState, action::AbstractDesign)
    return model(s.sol.total.u[1], s.design, action, length(s.sol.total) - 1)
end


function get_target_u(sol::WaveSol)
    return cat(sol.u[2:end]..., dims = ndims(first(sol.u)) + 1)
end

function loss(model::IncScWaveNet, s::WaveEnvState, action::AbstractDesign)
    u_inc_true = get_target_u(s.sol.incident)
    u_sc_true = get_target_u(s.sol.scattered)

    u_inc_pred, u_sc_pred = model(s, action)
    return sqrt(mse(u_inc_true, u_inc_pred)) + sqrt(mse(u_sc_true, u_sc_pred))
end

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

design_kwargs = Dict(:width => 1, :hight => 1, :spacing => 1.0f0, :c => 0.20f0, :center => [0.0f0, 0.0f0])
config = random_radii_scatterer_formation(;design_kwargs...)
grid_size = 4.0f0
elements = 128
fields = 6
dim = TwoDim(grid_size, elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

env = gpu(WaveEnv(
    # initial_condition = PlaneWave(dim, -2.0f0, 10.0f0),
    initial_condition = Pulse(dim, -3.0f0, 0.0f0, 10.0f0),
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
@time data = generate_episode_data(policy, env, 10)
z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))

model = gpu(IncScWaveNet(
        fields = fields, h_fields = 1, z_fields = 2, activation = relu,
        design_size = 2 * length(vec(config)), h_size = 64, grid_size = grid_size,
        z_elements = z_elements; dynamics_kwargs...))

opt = Adam(0.001)
ps = Flux.params(model)

train_loader = Flux.DataLoader(data, shuffle = true)

train_loss_history = Float32[]

for (s, a) in train_loader

    s, a = gpu(s[1]), gpu(a[1])

    gs = Flux.gradient(ps) do
        train_loss = loss(model, s, a)

        Flux.ignore() do
            push!(train_loss_history, train_loss)
            println(train_loss)
        end

        return train_loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, train_loss_history, linewidth = 3)
save("train_loss.png", fig)

# idx = 5
# u_inc_pred, u_sc_pred = cpu(model(s[idx], a[idx]))
# u_inc_true = get_target_u(s[idx].sol.incident) |> cpu
# u_sc_true = get_target_u(s[idx].sol.scattered) |> cpu
# plot_comparison!(u_inc_pred, u_sc_pred, u_inc_true, u_sc_true, 100, path = "u.png")

