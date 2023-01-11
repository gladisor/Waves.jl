# Waves.jl

## Usage

First install all dependancies using the instantiate comand in the REPL. To run a simulation execute the following code.

```
using Waves

sim = WaveSim(
    wave = Wave(dim = OneDim(-5.0, 5.0)),
    ic = GaussianPulse(1.0),
    t_max = 10.0,
    speed = 2.0,
    n = 30)

Waves.step!(sim)

```

Once a simulation has been run you can generate an animation using the render command.

```
render!(sim, path = "animations/1d.gif")
```

This simulator works for one, two, and three dimensional wave simulations.

Two Dimensions             |  One Dimension
:-------------------------:|:-------------------------:
![](https://github.com/gladisor/Waves.jl/blob/main/animations/2d.gif)  |  ![](https://github.com/gladisor/Waves.jl/blob/main/animations/1d.gif)