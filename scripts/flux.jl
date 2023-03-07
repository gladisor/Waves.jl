using Waves
using CairoMakie
using Flux
using Statistics: mean
using Flux.Optimisers: Restructure
Flux.CUDA.allowscalar(false)

struct WaveEncoder
    cell::WaveCell
    dynamics::WaveDynamics
    steps::Int
    layers::Chain
end

Flux.@functor WaveEncoder
Flux.trainable(encoder::WaveEncoder) = (encoder.layers,)

function (encoder::WaveEncoder)(wave::AbstractArray{Float32, 3})
    z = dropdims(encoder.layers(Flux.batch([wave])), dims = 3)
    latents = integrate(encoder.cell, z, encoder.dynamics, encoder.steps)
    latents = Flux.flatten(cat(latents..., dims = 3))
    return latents
end

function (encoder::WaveEncoder)(sol::WaveSol{TwoDim})
    return encoder(first(sol.u))
end

struct WaveDecoder
    dim::TwoDim
    points::AbstractMatrix{Float32}
    fields::Int
    restructure::Restructure
    hypernetwork::Chain
end

Flux.@functor WaveDecoder
Flux.trainable(decoder::WaveDecoder) = (decoder.hypernetwork,)

function WaveDecoder(dim::AbstractDim, in_size::Int, h_size::Int, fields::Int)

    points = reshape(grid(dim), :, 2)'

    Φ = Chain(
        Dense(ndims(dim), h_size, relu), 
        Dense(h_size, h_size, relu),
        Dense(h_size, h_size, relu),
        Dense(h_size, fields))

    _, restructure = Flux.destructure(Φ)

    hypernetwork = Chain(
        Dense(in_size, h_size, relu),
        Dense(h_size, h_size, relu), 
        Dense(h_size, length(restructure), bias = false))

    return WaveDecoder(dim, points, fields, restructure, hypernetwork)
end

function (decoder::WaveDecoder)(z)
    θ = decoder.hypernetwork(z)
    Φ = decoder.restructure.(eachcol(θ))
    return map(Φ -> reshape(Φ(decoder.points), size(decoder.dim)..., decoder.fields), Φ)
end

function training_data(sol::WaveSol, k::Int)
    x = []
    y = []

    for i ∈ 1:(length(sol) - k)
        push!(x, sol.u[i])
        push!(y, sol.u[i+1:i+k])
    end

    return (x, y)
end

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

dim = TwoDim(5.0f0, 0.1f0)
pulse = Pulse(dim, 0.0f0, 0.0f0, 5.0f0)
wave = zeros(Float32, size(dim)..., 6)
wave = pulse(wave) |> gpu

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 100.0f0, :ambient_speed => 2.0f0, :dt => 0.01f0)
cell = WaveCell(split_wave_pml, runge_kutta)

dynamics = WaveDynamics(dim = dim; dynamics_kwargs...) |> gpu

n = 200
@time u = integrate(cell, wave, dynamics, n)
pushfirst!(u, wave) ## add the initial state
t = collect(range(0.0f0, dynamics.dt * n, n + 1))
sol = WaveSol(dim, t, u) |> gpu

z_grid_size = 5.0f0
z_elements = 200
z_fields = 2

z_cell = WaveCell(latent_wave, runge_kutta)
z_dim = OneDim(z_grid_size, z_elements)
z_dynamics = WaveDynamics(dim = z_dim; dynamics_kwargs...) |> gpu

h_size = 12

layers = Chain(
    Conv((3, 3), 6 => h_size, relu),
    MaxPool((2, 2)),
    Conv((3, 3), h_size => h_size, relu),
    MaxPool((2, 2)),
    Conv((4, 4), h_size => 1, relu),
    Flux.flatten,
    z -> reshape(z, 200, 2, :)
)

decoder = Chain(
    z -> reshape(z, 20, 20, 1, size(z, 2)),
    Upsample(scale = (2, 2)),
    Conv((3, 3), 1 => h_size, relu),
    Conv((3, 3), h_size => h_size, relu),
    Conv((3, 3), h_size => h_size, relu),
    Upsample((2, 2)),
    Conv((5, 5), h_size => h_size, relu),
    Conv((5, 5), h_size => h_size, relu),
    Conv((5, 5), h_size => h_size, relu),
    Upsample((2, 2)),
    Conv((5, 5), h_size => h_size, relu),
    Conv((5, 5), h_size => h_size, relu),
    Conv((4, 4), h_size => 6)) |> gpu

encoder = WaveEncoder(z_cell, z_dynamics, length(sol)-1, layers) |> gpu
y_true = cat(sol.u[2:end]..., dims = 4)

ps = Flux.params(encoder, decoder)
opt = Adam(0.0005)

train_loss = Float32[]

for i ∈ 1:1000
    Waves.reset!(dynamics)
    
    gs = Flux.gradient(ps) do

        y_pred = decoder(encoder(sol))
        loss = sqrt(Flux.Losses.mse(y_pred, y_true))

        Flux.ignore() do 
            println("Loss: $loss")
            push!(train_loss, loss)
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end

y_pred = decoder(encoder(sol))

u_pred = [y_pred[:, :, :, i] for i in axes(y_pred, 4)]
pushfirst!(u_pred, wave)
sol_pred = cpu(WaveSol(dim, t, u_pred))
render!(sol_pred, path = "vid_pred.mp4")

plot_comparison!(cpu(y_true), cpu(y_pred), path = "rmse_comparison.png")

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, train_loss)
save("rmse_loss.png", fig)