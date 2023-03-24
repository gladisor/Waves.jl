using JLD2
using Flux
Flux.CUDA.allowscalar(false)
using Flux.Losses: mse, mae, msle
using CairoMakie

using Waves

struct Wave{D <: AbstractDim}
    u::AbstractArray{Float32}
end

Flux.@functor Wave

function Wave(dim::D, fields::Int = 1) where D <: AbstractDim
    u = zeros(Float32, size(dim)..., fields)
    return Wave{D}(u)
end

function Base.size(wave::Wave)
    return size(wave.u)
end

function Base.getindex(wave::Wave, idx...)
    return wave.u[idx...]
end

function Base.axes(wave::Wave, idx)
    return axes(wave.u, idx)
end

function energy(wave::Wave{OneDim})
    return sum(wave[:, 1] .^ 2)
end

function Waves.Pulse(dim::OneDim; x::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    return Pulse(dim, x, intensity)
end

function Waves.Pulse(dim::TwoDim; x::Float32 = 0.0f0, y::Float32 = 0.0f0, intensity::Float32 = 1.0f0)
    return Pulse(dim, x, y, intensity)
end

function (pulse::Pulse{OneDim})(wave::Wave{OneDim})
    u0 = exp.(-pulse.intensity * pulse.mesh_grid .^ 2)
    return Wave{OneDim}(hcat(u0, wave[:, 2:end]))
end

function (pulse::Pulse{TwoDim})(wave::Wave{TwoDim})
    u0 = dropdims(exp.(-pulse.intensity * sum(pulse.mesh_grid .^ 2, dims = 3)), dims = 3)
    return Wave{TwoDim}(cat(u0, wave[:, :, 2:end], dims = 3))
end

function Waves.split_wave_pml(wave::Wave{OneDim}, t::Float32, dynamics::WaveDynamics)
    u = wave[:, 1]
    v = wave[:, 2]
    ∇ = dynamics.grad

    ∇ = dynamics.grad
    σx = dynamics.pml
    C = Waves.speed(dynamics, t)
    b = C .^ 2

    du = b .* (∇ * v) .- σx .* u
    dvx = ∇ * u .- σx .* v

    return Wave{OneDim}(cat(du, dvx, dims = 2))
end

include("design_encoder.jl")
include("wave_net.jl")

file = jldopen("data/small/test/data5.jld2")

s = file["s"]
a = file["a"]

sol = s.sol.total
dim = sol.dim
elements = size(dim)[1]
grid_size = maximum(dim.x)
design = s.design

design_size = 2 * length(vec(design))
z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))

fields = size(sol.u[1], 3)
h_fields = 32
z_fields = 2
activation = relu

dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
dynamics = WaveDynamics(dim = OneDim(4.0f0, z_elements); dynamics_kwargs...)
cell = WaveCell(split_wave_pml, runge_kutta)

model = Chain(
    WaveEncoder(fields, h_fields, z_fields, activation),
    z -> integrate(cell, z, dynamics, length(sol) - 1),
    z -> cat(z..., dims = 3),
    z -> reshape(z, n, n, z_fields, :),
    UpBlock(3, z_fields,   h_fields, activation),
    UpBlock(3, h_fields,   h_fields, activation),
    UpBlock(3, h_fields,   h_fields,   activation),
    Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
    Conv((3, 3), h_fields => h_fields, activation, pad = SamePad()),
    Conv((3, 3), h_fields => 1, tanh, pad = SamePad()),
    z -> dropdims(z, dims = 3)
) |> gpu

sol = gpu(sol)
u_true = get_target_u(sol)
u_true = u_true[:, :, 1, :]
n = Int(sqrt(z_elements))

opt = Adam(0.0001)
ps = Flux.params(model)

train_loss = Float32[]

for i in 1:1000

    Waves.reset!(dynamics)

    gs = Flux.gradient(ps) do

        u_pred = model(sol)
        loss = sqrt(mse(u_true, u_pred))

        Flux.ignore() do

            println(loss)
            push!(train_loss, loss)

            if i % 5 == 0
                fig = Figure()
                ax1 = Axis(fig[1, 1], aspect = 1.0, title = "True Scattered Wave", xlabel = "X (m)", ylabel = "Y(m)")
                ax2 = Axis(fig[1, 2], aspect = 1.0, title = "Predicted Scattered Wave", xlabel = "X (m)", yticklabelsvisible = false)
                heatmap!(ax1, dim.x, dim.y, cpu(u_true[:, :, 50]), colormap = :ice)
                heatmap!(ax2, dim.x, dim.y, cpu(u_pred[:, :, 50]), colormap = :ice)
                save("comparison.png", fig)

                fig = Figure()
                ax = Axis(fig[1, 1])
                lines!(ax, train_loss)
                save("loss.png", fig)
            end

        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end
