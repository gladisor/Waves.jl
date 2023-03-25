using JLD2
using Flux
Flux.CUDA.allowscalar(false)
using Flux: DataLoader
using Flux.Losses: mse, huber_loss
using CairoMakie
using Statistics: mean

using Waves

include("design_encoder.jl")
include("wave_net.jl")

function load_wave_data(path::String)
    file = jldopen(path)
    s = file["s"]
    a = file["a"]
    return (s, a)
end

function Waves.energy(sol::WaveSol)
    return sum.(energy.(displacement.(sol.u)))
end

struct WaveDesignEncoder
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
end

Flux.@functor WaveDesignEncoder

function (encoder::WaveDesignEncoder)(sol::WaveSol, design::AbstractDesign, action::AbstractDesign)
    z = encoder.wave_encoder(sol)
    b = encoder.design_encoder(design, action)
    return hcat(z, b)
end

struct SigmaControlModel
    wave_encoder::WaveEncoder
    design_encoder::DesignEncoder
    z_cell::WaveCell
    z_dynamics::WaveDynamics
    mlp::Chain
end

Flux.@functor SigmaControlModel (wave_encoder, design_encoder, mlp)

function (model::SigmaControlModel)(sol::WaveSol, design::AbstractDesign, action::AbstractDesign)
    z = model.wave_encoder(sol)
    b = model.design_encoder(design, action)
    latents = cat(integrate(model.z_cell, hcat(z, b), model.z_dynamics, length(sol) - 1)..., dims = 3)
    return vec(model.mlp(Flux.flatten(latents)))
end

data = load_wave_data.(readdir("data/small/train", join = true))
s, a = first(data)

dim = s.sol.total.dim
elements = size(dim)[1]
grid_size = maximum(dim.x)

z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))
fields = size(s.sol.total.u[1], 3)
h_fields = 64
z_fields = 2
activation = relu
h_size = 128

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
dynamics = WaveDynamics(dim = OneDim(4.0f0, z_elements); dynamics_kwargs...) |> gpu
n = Int(sqrt(z_elements))
cell = WaveCell(split_wave_pml, runge_kutta)

sol_total = gpu(s.sol.total)
design = gpu(s.design)
a = gpu(a)

model = SigmaControlModel(
    WaveEncoder(fields, h_fields, z_fields, activation),
    DesignEncoder(2 * length(vec(design)), h_size, z_elements, activation),
    WaveCell(nonlinear_latent_wave, runge_kutta),
    dynamics,
    Chain(
        Dense(3 * z_elements, h_size, activation),
        Dense(h_size, 1),
    )
) |> gpu

opt = Adam(0.0009)
ps = Flux.params(model)
train_loss = Float32[]
train_loader = DataLoader(data[1:2], shuffle = true)

for epoch in 1:100

    epoch_loss = Float32[]
    # for (i, (s, a)) in enumerate(train_loader)

    #     s, a = s[1], a[1]

        sol_total = gpu(s.sol.total)
        sol_scattered = gpu(s.sol.scattered)
        sigma_true = gpu(energy(sol_scattered)[2:end])
        Waves.reset!(dynamics)

        gs = Flux.gradient(ps) do
            sigma_pred = model(sol_total, design, )
            loss = huber_loss(sigma_true, sigma_pred)

            Flux.ignore() do
                println("Loss: $loss")
                push!(epoch_loss, loss)
            end

            return loss
        end

        Flux.Optimise.update!(opt, ps, gs)
    # end

    push!(train_loss, mean(epoch_loss))

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, train_loss)
    save("loss.png", fig)
end