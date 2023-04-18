using Flux
using Flux: Recur, pullback
using ChainRulesCore

using Interpolations
using Interpolations: Extrapolation
using CairoMakie
using Optimisers
using Waves

include("../src/dynamics.jl")
include("plot.jl")

grid_size = 8.0f0
ambient_speed = 343.0f0
ti =  0.0f0
dt = 0.00005f0
steps = 100

latent_dim = OneDim(grid_size, 1024)
latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 10000.0f0)
iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)

pulse = Pulse(latent_dim, 0.0f0, 1.0f0)
zi = pulse(build_wave(latent_dim, fields = 3))
zi[:, 3] .= 1.0f0

opt_state = Optimisers.setup(Optimisers.Momentum(1e-4), iter)
# opt_state = Optimisers.setup(Optimisers.Adam(1e-4), iter)

for i in 1:50
    loss, back = pullback(_iter -> Flux.mean(sum(_iter(zi)[:, 1, :] .^ 2, dims = 1)), iter)
    gs = back(one(loss))[1]
    opt_state, iter = Optimisers.update(opt_state, iter, gs)
    println(loss)
end

render!(latent_dim, iter(zi), path = "vid.mp4")

# ### MPC COST
# # s = gpu(state(env))
# # control_sequence = gpu(build_control_sequence(initial_design, 2))

# # opt_state = Optimisers.setup(Optimisers.Momentum(1e-5), control_sequence)

# # for i in 1:10
# #     cost, back = pullback(a -> build_mpc_cost(model, s, a), control_sequence)
# #     gs = back(one(cost))[1]
# #     opt_state, control_sequence = Optimisers.update(opt_state, control_sequence, gs)
# #     println(cost)
# # end