using ProgressMeter
using Distributions

using Waves
include("model.jl")

wave = Wave(dim = OneDim(-10.0, 10.0))
t_max = 10.0
speed = 2.0
n = 50
dt = 0.05

data = Vector{WaveSol{OneDim}}()
loc = Uniform(-7.5, 7.5)

@showprogress for i ∈ 1:10
    pulse = GaussianPulse(intensity = 1.0, loc = [rand(loc)])
    sim = WaveSim(wave = wave, ic = pulse, t_max = t_max, speed = speed, n = n, dt = dt)
    Waves.step!(sim)
    sol = WaveSol(sim)
    push!(data, sol)
end

pulse = GaussianPulse(intensity = 1.0)
sim = WaveSim(wave = wave, ic = pulse, t_max = t_max, speed = speed, n = n, dt = dt)
model = WaveNet(sim,
    in_size = 51,
    enc_h_size = 64, 
    enc_n_hidden = 1, 
    z_size = 2,
    Φ_h_size = 64, Φ_n_hidden = 1,
    Ψ_h_size = 64, 
    Ψ_n_hidden = 1, 
    σ = relu)

