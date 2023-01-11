using Waves

design = ParameterizedDesign(Cylinder(-5.0, 0.0))

sim = WaveSim(
    wave = Wave(dim = OneDim(-5.0, 5.0)),
    ic = GaussianPulse(1.0),
    t_max = 10.0,
    speed = 2.0,
    n = 30)

Waves.step!(sim)
render!(sim, path = "animations/1d.gif")