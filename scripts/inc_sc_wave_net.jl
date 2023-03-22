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

function Flux.gpu(model::IncScWaveNet)
    return IncScWaveNet(
        gpu(model.wave_encoder),
        gpu(model.design_encoder),
        model.z_cell,
        gpu(model.z_dynamics),
        gpu(model.inc_decoder),
        gpu(model.sc_decoder)
    )
end

function Flux.cpu(model::IncScWaveNet)
    return IncScWaveNet(
        cpu(model.wave_encoder),
        cpu(model.design_encoder),
        model.z_cell,
        cpu(model.z_dynamics),
        cpu(model.inc_decoder),
        cpu(model.sc_decoder)
    )
end

function loss(model::IncScWaveNet, s::WaveEnvState, action::AbstractDesign)
    u_inc_true = get_target_u(s.sol.incident)
    u_sc_true = get_target_u(s.sol.scattered)
    u_inc_pred, u_sc_pred = model(s, action)
    return sqrt(mse(u_inc_true, u_inc_pred)) + sqrt(mse(u_sc_true, u_sc_pred))
end

function Waves.reset!(model::IncScWaveNet)
    Waves.reset!(model.z_dynamics)
end

# function Waves.plot_comparison!(
#         u_inc_pred::AbstractArray, u_sc_pred::AbstractArray, 
#         u_inc_true::AbstractArray, u_sc_true::AbstractArray, 
#         idx::Int; path::String)

#     fig = Figure()
#     ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Incident Wave", ylabel = "Y (m)", xticklabelsvisible = false, xticksvisible = false)
#     ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Incident Wave", yticklabelsvisible = false, yticksvisible = false, xticklabelsvisible = false, xticksvisible = false)
#     ax3 = Axis(fig[2, 1], aspect = 1.0, title = "True Scattered Wave", xlabel = "X (m)", ylabel = "Y (m)")
#     ax4 = Axis(fig[2, 2], aspect = 1.0, title = "Predicted Scattered Wave", xlabel = "X (m)", yticklabelsvisible = false, yticksvisible = false)

#     heatmap!(ax1, dim.x, dim.y, u_inc_true[:, :, 1, idx], colormap = :ice)
#     heatmap!(ax2, dim.x, dim.y, u_inc_pred[:, :, 1, idx], colormap = :ice)
#     heatmap!(ax3, dim.x, dim.y, u_sc_true[:, :, 1, idx], colormap = :ice)
#     heatmap!(ax4, dim.x, dim.y, u_sc_pred[:, :, 1, idx], colormap = :ice)

#     save(path, fig)
#     return nothing
# end

function Waves.plot_comparison!(model::IncScWaveNet, s::WaveEnvState, a::AbstractDesign; path::String)
    u_inc_pred, u_sc_pred = cpu(model(s, a))
    u_inc_true = get_target_u(s.sol.incident) |> cpu
    u_sc_true = get_target_u(s.sol.scattered) |> cpu
    # plot_comparison!(u_inc_pred, u_sc_pred, u_inc_true, u_sc_true, length(s.sol.total)-1, path = path)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Incident Wave", ylabel = "Y (m)", xticklabelsvisible = false, xticksvisible = false)
    ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Incident Wave", yticklabelsvisible = false, yticksvisible = false, xticklabelsvisible = false, xticksvisible = false)
    ax3 = Axis(fig[2, 1], aspect = 1.0, title = "True Scattered Wave", xlabel = "X (m)", ylabel = "Y (m)")
    ax4 = Axis(fig[2, 2], aspect = 1.0, title = "Predicted Scattered Wave", xlabel = "X (m)", yticklabelsvisible = false, yticksvisible = false)

    dim = s.sol.total.dim
    idx = size(u_sc_pred, 4)

    heatmap!(ax1, dim.x, dim.y, u_inc_true[:, :, 1, idx], colormap = :ice)
    heatmap!(ax2, dim.x, dim.y, u_inc_pred[:, :, 1, idx], colormap = :ice)
    heatmap!(ax3, dim.x, dim.y, u_sc_true[:, :, 1, idx], colormap = :ice)
    heatmap!(ax4, dim.x, dim.y, u_sc_pred[:, :, 1, idx], colormap = :ice)

    save(path, fig)
    return nothing
end
