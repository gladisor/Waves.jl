struct WaveNet
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    z_cell::AbstractWaveCell
    z_dynamics::WaveDynamics
    wave_decoder::WaveDecoder
end

Flux.@functor WaveNet (wave_encoder, design_encoder, wave_decoder)

function WaveNet(;fields::Int, h_fields::Int, z_fields::Int, activation::Function, design_size::Int, h_size::Int, grid_size::Float32, z_elements::Int, dynamics_kwargs...)
    wave_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
    design_encoder = DesignEncoder(design_size, h_size, z_elements, activation)
    z_cell = WaveCell(nonlinear_latent_wave, runge_kutta)
    z_dynamics = WaveDynamics(dim = OneDim(grid_size, z_elements); dynamics_kwargs...)
    wave_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation)
    return WaveNet(wave_encoder, design_encoder, z_cell, z_dynamics, wave_decoder)
end

function (model::WaveNet)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign, steps::Int)
    z = hcat(model.wave_encoder(wave), model.design_encoder(design, action))
    latents = integrate(model.z_cell, z, model.z_dynamics, steps)
    z_concat = cat(latents..., dims = 3)
    n = Int(sqrt(size(z_concat, 1)))
    z_feature_maps = reshape(z_concat, n, n, size(z_concat, 2), :)
    return model.wave_decoder(z_feature_maps)
end

function (model::WaveNet)(s::WaveEnvState, action::AbstractDesign)
    return model(s.sol.total.u[1], s.design, action, length(s.sol.total) - 1)
end

function Flux.gpu(model::WaveNet)
    return WaveNet(
        gpu(model.wave_encoder),
        gpu(model.design_encoder),
        model.z_cell,
        gpu(model.z_dynamics),
        gpu(model.wave_decoder),
    )
end

function Flux.cpu(model::WaveNet)
    return WaveNet(
        cpu(model.wave_encoder),
        cpu(model.design_encoder),
        model.z_cell,
        cpu(model.z_dynamics),
        cpu(model.wave_decoder),
    )
end

function loss(model::WaveNet, s::WaveEnvState, action::AbstractDesign)
    # u_true = get_target_u(s.sol.total)
    u_true = get_target_u(s.sol.scattered)
    u_pred = model(s, action)
    return sqrt(mse(u_true, u_pred))
end

function Waves.reset!(model::WaveNet)
    Waves.reset!(model.z_dynamics)
end

# function Waves.plot_comparison!(u_pred::AbstractArray{Float32, 4}, u_true::AbstractArray{Float32, 4}, idx::Int; path::String)
#     fig = Figure()
#     ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Total Wave", xlabel = "X (m)", ylabel = "Y(m)")
#     ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Total Wave", xlabel = "X (m)", yticklabelsvisible = false)

#     heatmap!(ax1, dim.x, dim.y, u_true[:, :, 1, idx], colormap = :ice)
#     heatmap!(ax2, dim.x, dim.y, u_pred[:, :, 1, idx], colormap = :ice)

#     save(path, fig)
#     return nothing
# end

function Waves.plot_comparison!(model::WaveNet, s::WaveEnvState, a::AbstractDesign; path::String)
    u_pred = cpu(model(s, a))
    u_true = get_target_u(s.sol.scattered) |> cpu
    # u_true = get_target_u(s.sol.total) |> cpu
    # plot_comparison!(u_pred, u_true, length(s.sol.total) - 1, path = path)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Scattered Wave", xlabel = "X (m)", ylabel = "Y(m)")
    ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Scattered Wave", xlabel = "X (m)", yticklabelsvisible = false)

    dim = cpu(s.sol.total.dim)
    idx = size(u_pred, 4)
    heatmap!(ax1, dim.x, dim.y, u_true[:, :, 1, idx], colormap = :ice)
    heatmap!(ax2, dim.x, dim.y, u_pred[:, :, 1, idx], colormap = :ice)

    save(path, fig)
    return nothing
end