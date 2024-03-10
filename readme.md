# Waves.jl

This package implements a "gym" like environment for exploring interaction between controllers and acoustic Partial Differential Equations (PDE). It provides an environment which simulates the acoustic wave equation in 2D space with a perfectly matched layer (PML). 

## Demo

This video shows a model predictive controller manipulating a design to surpress scattered energy in the environment.

<p align="center">
	<img src="https://github.com/gladisor/Waves.jl/blob/wildfire/images/mpc.gif">
</p>

Here the physically informed latent space of our model is shown at the same time as the state of the real environment. The energy of the scattered field is also visualized. It is demonstrated that the energy of the scattered latent field matches the scattered energy in the real environment.

<p align="center">
	<img src="https://github.com/gladisor/Waves.jl/blob/wildfire/images/dashboard.gif">
</p>

## Usage

First import the package and Flux:
```
using Waves, Flux
```

The environment can be instantiated through its constructor. However several precursor components must be defined.

The finite element grid is defined:
```
dim = TwoDim(15.0f0, 700)
```

The design space is also needed. In this example a ring configuration of scatterers is selected.
```
design_space = Waves.build_triple_ring_design_space()
```

The source of acoustic energy is also needed. One which randomly resets to a different location each episode is instantiated:
```
μ_low = [-10.0f0 -10.0f0]
μ_high = [-10.0f0 10.0f0]
σ = [0.3f0]
a = [1.0f0]
source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
```

Finally the environment is built and sent to the gpu:
```
env = gpu(sWaveEnv(dim; design_space, source, integration_steps = 100, actions = 20))
```