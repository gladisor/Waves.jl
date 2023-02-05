using Waves

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

# design = Design(Configuration([0.0], [-3.0], [0.5], [0.0]))
kwargs = Dict(:wave => wave, :ic => GaussianPulse(1.0), :boundary => OpenBoundary(), :ambient_speed => 1.0, :tmax => 20.0, :n => 21, :dt => 0.1)