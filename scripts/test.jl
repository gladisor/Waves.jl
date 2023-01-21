using Waves
import ModelingToolkit

function plane_wave(wave::Wave{TwoDim})
    _, y, t, u = Waves.unpack(wave)
    x_min, _ = ModelingToolkit.getbounds(wave.dim.x)
    bcs = open_boundary(wave)
    bcs[1] = u(x_min, y, t) ~ exp(-10*(t-1.1)^2)
    return bcs
end

gs = 5.0
dim = TwoDim(-gs, gs, -gs, gs)
wave = Wave(dim = dim)

@time sim = WaveSim(
    wave = wave, 
    # ic = GaussianPulse(1.0), 
    ic = Silence(),
    boundary = plane_wave,
    ambient_speed = 2.0, tmax = 10.0, n = 21, dt = 0.05)

@time Waves.propagate!(sim)
@time sol = WaveSol(sim)
@time render!(sol, path = "vid.mp4")
@time plot_energy!(sol_inc = sol, path = "inc.png")