using Flux

Base.@kwdef struct Archetecture
    in_size::Int
    out_size::Int 
    h_size::Int
    n_hidden::Int = 1
    σ = relu
    normalization = identity
end

struct SequenceEncoder
    layers::Chain
end

Flux.@functor SequenceEncoder

function SequenceEncoder(dim::OneDim, arch::Archetecture)
    layers = [LSTM(arch.in_size, arch.h_size)]

    for i ∈ 1:arch.n_hidden
        push!(layers, LSTM(arch.h_size, arch.h_size))
    end

    push!(layers, LSTM(arch.h_size, arch.out_size))

    return SequenceEncoder(Chain(layers))
end

function (model::SequenceEncoder)(x)
    y = model.layers(x)
    return y[:, end, :]
end

# gs = 5.0
# dim = OneDim(size = gs)
# wave = Wave(dim = dim)
# kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 100, :dt => 0.05)
# sim = WaveSim(;kwargs...)
# propagate!(sim)
# sol = WaveSol(sim)

function sequence(x::Vector, n::Int, k::Int)
    context = []
    target = []
    for i ∈ n:(length(x) - k)
        push!(context, x[(i-n + 1):i])
        push!(target, x[i+k])
    end

    return context, target
end

# x = collect(1:length(sol))
x = sol.data
c, t = sequence(x, 2, 3)

# arch = Archetecture(in_size = 10, h_size = 128, out_size = 2)
# model = SequenceEncoder(wave.dim, arch)
# # (features, sequence, batch)
# x = randn(Float32, 10, 5, 32)
# model(x)