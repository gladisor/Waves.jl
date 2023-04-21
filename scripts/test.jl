include("dependencies.jl")

random_design_kwargs = Dict(:width => 1, :hight => 2, :spacing => 3.0f0, :r => Waves.MAX_RADII, :c => 2100.0f0, :center => [0.0f0, 0.0f0])
## load env from disk
env = gpu(BSON.load("results/radii/PercentageWaveControlModel_old/env.bson")[:env])
reset!(env)

## model dependencies
latent_elements = 64
latent_dim = OneDim(grid_size, latent_elements)
latent_dynamics = LatentPMLWaveDynamics(latent_dim, ambient_speed = ambient_speed, pml_scale = 5000.0f0)
design_size = 2 * length(vec(rand(action_space(env))))

## wave
wave_encoder = WaveEncoder(6, 8, 1, tanh)
wave_encoder_mlp = Chain(Dense(1024, latent_elements, tanh), vec)

## design & action
design_encoder = DesignEncoder(design_size, latent_elements, latent_elements, tanh)
design_encoder_mlp = Chain(c -> c .+ 0.5f0)

## latent integration
iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)
mlp = Chain(Flux.flatten, Dense(3 * latent_elements, latent_elements, relu), Dense(latent_elements, latent_elements, relu), Dense(latent_elements, 1), vec)

## instantiate model
model = gpu(PercentageWaveControlModel(wave_encoder, wave_encoder_mlp, design_encoder, design_encoder_mlp, iter, mlp))

## random policy
policy = RandomDesignPolicy(action_space(env))
reset!(env)

s = state(env)
zi = encode(model, gpu(s.wave_total), gpu(s.design), gpu(policy(env)))

cost, back = pullback(_zi -> mean(sum(model.iter(_zi)[:, 1, :] .^ 2, dims = 1)), zi)
gs = cpu(back(one(cost))[1])

fig = Figure()
ax1 = Axis(fig[1, 1], aspect = 1.0f0, title = "Displacement", ylabel = "Gradient Value", xlabel = "Distance ?")
ax2 = Axis(fig[1, 2], aspect = 1.0f0, title = "Velocity", xlabel = "Distance ?")
ax3 = Axis(fig[1, 3], aspect = 1.0f0, title = "Speed", xlabel = "Distance ?")

lines!(ax1, latent_dim.x, gs[:, 1])
lines!(ax2, latent_dim.x, gs[:, 2])
lines!(ax3, latent_dim.x, gs[:, 3])
save("gs.png", fig)

# ## propagate a few random actions
# for i in 1:2
#     env(policy(env))
# end

# s = state(env)
# z = model.iter(encode(model, gpu(s.wave_total), gpu(s.design), gpu(policy(env))))
# render!(latent_dim, cpu(z), path = "latent_wave.mp4")

# fig = Figure()
# ax = Axis(fig[1, 1], aspect = 1.0f0)
# s = state(env)

# for i in 1:4
#     sigma_pred = cpu(model(s, gpu(policy(env))))
#     lines!(ax, tspan, sigma_pred)
# end

# save("untrained_model.png", fig)