# Waves.jl

## Usage

First install all dependancies using the instantiate comand in the REPL. To run a simulation execute the following code.

```
using Waves


grid_size = 5.0

wave = Wave(dim = OneDim(-grid_size, grid_size))
pulse = GaussianPulse(intensity = 1.0)
sim = WaveSim(wave = wave, ic = pulse, t_max = 10.0, speed = 1.0, n = 30, dt = 0.05)
Waves.step!(sim)
sol = WaveSol(sim)
```

Once a simulation has been run you can generate an animation using the render command.

```
render!(sol, path = "1d.gif")
```

This simulator works for one, two, and three dimensional wave simulations. In order to use a higher dimension change the dim argument in the Wave constructor.

Two Dimensions             |  One Dimension
:-------------------------:|:-------------------------:
![](https://github.com/gladisor/Waves.jl/blob/main/animations/2d.gif)  |  ![](https://github.com/gladisor/Waves.jl/blob/main/animations/1d.gif)

The goal of this project is to develop an environment in which Reinforcement Learning agents can learn to control acoustic waves. We have created an object which allows for interactive control with the wave simulator through applying actions to a design. An example of random control in this environment is shown below.

```
wave = Wave(dim = TwoDim(-grid_size, grid_size, -grid_size, grid_size))
design = ParameterizedDesign(Cylinder(-3.0, -3.0, 0.5, 0.0))
pulse = GaussianPulse(intensity = 5.0, loc = [2.5, 2.5])

sim = WaveSim(wave = wave, design = design, ic = pulse, t_max = 10.0, speed = 1.0, n = 30, dt = 0.05)

env = WaveEnv(sim, design, 20)

reset!(env)
env.design = ParameterizedDesign(Cylinder(0.0, 0.0, 0.5, 0.2))
steps = Vector{typeof(env.design.design)}([env.design.design])

while !is_terminated(env)
    action = Cylinder(randn() * 0.5, randn() * 0.5, 0.0, 0.0)
    [push!(steps, s) for s ∈ step(env, action)]
end

sol = WaveSol(env.sim)
steps = vcat(steps...)
render!(sol, design = steps, path = "env.mp4")
```

![](https://github.com/gladisor/Waves.jl/blob/main/animations/active.gif)