# Waves.jl

## Usage

First install all dependancies using the instantiate comand in the REPL. To run a simulation execute the following code.

```
using Waves

sim = WaveSim(
    wave = Wave(dim = OneDim(-10.0, 10.0)),
    ic = GaussianPulse(intensity = 1.0),
    t_max = 10.0,
    speed = 2.0,
    n = 30,
    dt = 0.05)

Waves.step!(sim)
sol = WaveSol(sim)
```

Once a simulation has been run you can generate an animation using the render command.

```
render!(sol, path = "animations/1d.gif")
```

This simulator works for one, two, and three dimensional wave simulations. In order to use a higher dimension change the dim argument in the Wave constructor.

Two Dimensions             |  One Dimension
:-------------------------:|:-------------------------:
![](https://github.com/gladisor/Waves.jl/blob/main/animations/2d.gif)  |  ![](https://github.com/gladisor/Waves.jl/blob/main/animations/1d.gif)