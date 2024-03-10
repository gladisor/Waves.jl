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