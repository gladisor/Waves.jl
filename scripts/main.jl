using JLD2
using Flux
using Flux.Losses: mse
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

# elements = 256
# dim = OneDim(4.0f0, elements)
# dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)
# dynamics = WaveDynamics(dim = dim; dynamics_kwargs...)

# fields = 2
# wave = Wave(dim, fields)
# pulse = Pulse(dim, intensity = 10.0f0)
# wave = pulse(wave)

# model = Chain(
#     w -> w.u,
#     Dense(elements, 10, relu),
#     x -> Wave{OneDim}(x))

# opt = Descent(0.01)
# ps = Flux.params(wave)

# gs = Flux.gradient(ps) do 
#     return energy(model(wave))
# end

function load_wave_data(path::String)

    s = WaveEnvState[]
    a = AbstractDesign[]

    for file_path in readdir(path, join = true)
        jldopen(file_path) do file
            println(file)
            push!(s, file["s"])
            push!(a, file["a"])
        end
    end

    return (s, a)
end

include("design_encoder.jl")
include("wave_net.jl")

file = jldopen("data/small/train/data5.jld2")

s = file["s"]
a = file["a"]

dim = s.sol.total.dim
elements = size(dim)[1]
grid_size = maximum(dim.x)
design = s.design

design_size = 2 * length(vec(design))
z_elements = prod(Int.(size(dim) ./ (2 ^ 3)))

model_kwargs = Dict(:fields => 6, :h_fields => 128, :z_fields => 2, :activation => relu, :design_size => design_size, :h_size => 256, :grid_size => 4.0f0, :z_elements => z_elements)
dynamics_kwargs = Dict(:pml_width => 1.0f0, :pml_scale => 70.0f0, :ambient_speed => 1.0f0, :dt => 0.01f0)

dynamics = WaveDynamics(dim = OneDim(4.0f0, z_elements); dynamics_kwargs...) |> gpu
wave_encoder = WaveEncoder(model_kwargs[:fields], model_kwargs[:h_fields], model_kwargs[:z_fields], model_kwargs[:activation]) |> gpu
wave_decoder = WaveDecoder(1, model_kwargs[:h_fields], model_kwargs[:z_fields], model_kwargs[:activation]) |> gpu
cell = WaveCell(split_wave_pml, runge_kutta)

u_true = get_target_u(s.sol.total) |> gpu
u_true = u_true[:, :, 1, :]
n = Int(sqrt(z_elements))

opt = Adam(0.001)
ps = Flux.params(wave_encoder, wave_decoder)

for i in 1:10
    gs = Flux.gradient(ps) do

        z = wave_encoder(s.sol.total)
        z_sol = cat(integrate(cell, z, dynamics, length(s.sol.total) - 1)..., dims = ndims(z) + 1)
        z_feature_maps = reshape(z_sol, n, n, size(z_sol, 2), :)
        u_pred = dropdims(wave_decoder(z_feature_maps), dims = 3)
        loss = sqrt(mse(u_true, u_pred))

        Flux.ignore() do 
            println(loss)
        end

        return loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end
