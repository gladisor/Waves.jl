struct NonAutonomusCell <: AbstractWaveCell
    derivative_function::Function
    integration_function::Function
end

Flux.@functor NonAutonomusCell

function (cell::NonAutonomusCell)(z::AbstractArray{Float32}, dynamics::AbstractDynamics)
    z′ = z .+ cell.integration_function(cell.derivative_function, z, dynamics)
    return z′, z′
end

struct LatentWaveSeparation
    incident_encoder::WaveEncoder
    total_encoder::WaveEncoder
    design_encoder::DesignEncoder
    z_cell::NonAutonomusCell
    z_dynamics::WaveDynamics
    scattered_decoder::WaveDecoder
end

Flux.@functor LatentWaveSeparation

function LatentWaveSeparation(;fields::Int, h_fields::Int, z_fields::Int, activation::Function, design_size::Int, h_size::Int, grid_size::Float32, z_elements::Int, dynamics_kwargs...)
    incident_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
    total_encoder = WaveEncoder(fields, h_fields, z_fields, activation)
    design_encoder = DesignEncoder(design_size, 128, 1024, activation)
    z_cell = NonAutonomusCell(nonlinear_latent_wave, runge_kutta)
    z_dynamics = WaveDynamics(dim = OneDim(grid_size, z_elements); dynamics_kwargs...)
    scattered_decoder = WaveDecoder(fields, h_fields, z_fields + 1, activation)
    return LatentWaveSeparation(incident_encoder, total_encoder, design_encoder, z_cell, z_dynamics, scattered_decoder)
end

function (model::LatentWaveSeparation)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign, steps::Int)
    b = gpu(ones(Float32, size(model.z_dynamics.dim)[1]))
    z_inc = hcat(model.incident_encoder(wave), b)
    z_tot = hcat(model.total_encoder(wave), model.design_encoder(design, action))

    z_inc_sol = cat(integrate(model.z_cell, z_inc, model.z_dynamics, steps)..., dims = 3)
    z_tot_sol = cat(integrate(model.z_cell, z_tot, model.z_dynamics, steps)..., dims = 3)
    z_sc_sol = z_tot_sol .- z_inc_sol

    n = Int(sqrt(size(z_sc_sol, 1)))
    z_sc_feature_map = reshape(z_sc_sol, n, n, size(z_sc_sol, 2), :)
    return model.scattered_decoder(z_sc_feature_map)
end

function (model::LatentWaveSeparation)(s::WaveEnvState, action::AbstractDesign)
    return model(s.sol.total.u[1], s.design, action, length(s.sol.total) - 1)
end

function Flux.gpu(model::LatentWaveSeparation)
    return LatentWaveSeparation(
        gpu(model.incident_encoder),
        gpu(model.total_encoder),
        gpu(model.design_encoder),
        model.z_cell,
        gpu(model.z_dynamics),
        gpu(model.scattered_decoder)
    )
end

function Flux.cpu(model::LatentWaveSeparation)
    return LatentWaveSeparation(
        cpu(model.incident_encoder),
        cpu(model.total_encoder),
        cpu(model.design_encoder),
        model.z_cell,
        cpu(model.z_dynamics),
        cpu(model.scattered_decoder)
    )
end

function loss(model::LatentWaveSeparation, s::WaveEnvState, action::AbstractDesign)
    u_scattered_true = get_target_u(s.sol.scattered)
    u_scattered_pred = model(s, action)
    return sqrt(mse(u_scattered_true, u_scattered_pred))
end

function Waves.reset!(model::LatentWaveSeparation)
    return nothing
end

function Waves.plot_comparison!(model::LatentWaveSeparation, s::WaveEnvState, action::AbstractDesign; path::String)
    u_sc_pred = cpu(model(s, action))
    u_sc_true = cpu(get_target_u(s.sol.scattered))

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Scattered Wave", xlabel = "X (m)", ylabel = "Y(m)")
    ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Scattered Wave", xlabel = "X (m)", yticklabelsvisible = false)

    dim = s.sol.total.dim
    idx = size(u_sc_pred, 4)
    heatmap!(ax1, dim.x, dim.y, u_sc_true[:, :, 1, idx], colormap = :ice)
    heatmap!(ax2, dim.x, dim.y, u_sc_pred[:, :, 1, idx], colormap = :ice)

    save(path, fig)
    return nothing
end

# include("design_encoder.jl")
# file = jldopen("data/train/data15.jld2")

# s = file["s"]
# design = s.design
# action = file["a"]

# z_elements = prod(Int.(size(s.sol.total.dim) ./ (2 ^ 3)))
# design_size = length(vec(design)) * 2

# model_kwargs = Dict(:fields => 6, :h_fields => 128, :z_fields => 2, :activation => relu, :design_size => design_size, :h_size => 256, :grid_size => 4.0f0, :z_elements => z_elements)
# dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

# model = LatentWaveSeparation(;model_kwargs..., dynamics_kwargs...)

# plot_comparison!(model, s, action, path = "comparison.png")
# display(loss(model, s, action))