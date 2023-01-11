using ProgressMeter
using Distributions

using Waves

data = Vector{WaveSol{OneDim}}()

loc = Uniform(-7.5, 7.5)
@showprogress for i ∈ 1:10
    sim = WaveSim(
        wave = Wave(dim = OneDim(-10.0, 10.0)),
        ic = GaussianPulse(intensity = 1.0, loc = [rand(loc)]),
        t_max = 10.0,
        speed = 2.0,
        n = 30)

    Waves.step!(sim)
    sol = WaveSol(sim)
    push!(data, sol)
end

for (i, sol) ∈ enumerate(data)
    render!(sol, path = "vid$i.mp4")
end