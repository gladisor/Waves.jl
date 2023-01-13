using ProgressMeter
using Distributions

using Waves
include("model.jl")

wave = Wave(dim = OneDim(-10.0, 10.0))
t_max = 10.0
speed = 2.0
n = 50
dt = 0.05

# data = Vector{WaveSol{OneDim}}()
# loc = Uniform(-7.5, 7.5)

# @showprogress for i ∈ 1:2
#     pulse = GaussianPulse(intensity = 1.0, loc = [rand(loc)])
#     sim = WaveSim(wave = wave, ic = pulse, t_max = t_max, speed = speed, n = n, dt = dt)
#     Waves.step!(sim)
#     sol = WaveSol(sim)
#     push!(data, sol)
# end

pulse = GaussianPulse(intensity = 1.0)
sim = WaveSim(wave = wave, ic = pulse, t_max = t_max, speed = speed, n = n, dt = dt)

function sequence_data(sol::WaveSol{OneDim})::Matrix{Float32}
    return hcat(Waves.get_data(sol)...)
end

u = sequence_data.(data)

z_size = 3
encoder = Chain(LSTM(50, z_size))
emb = encoder(u[1])

"""
- How to do negative sampling? Choose samples from p(x) distribution to contrast with p(x | c) distribution
- Use every possible combonation of context and n step in the future?
- restrict lstm context to a rolling window? Or the entire series?
"""

