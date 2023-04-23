include("dependencies.jl")

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

## propagate a few random actions
for i in 1:2
    env(policy(env))
end

s = state(env)
z = model.iter(encode(model, gpu(s.wave_total), gpu(s.design), gpu(policy(env))))
render!(latent_dim, cpu(z), path = "latent_wave.mp4")
tspan = build_tspan(time(env), env.dt, env.integration_steps)
fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0f0)
s = state(env)

for i in 1:4
    sigma_pred = cpu(model(s, gpu(policy(env))))
    lines!(ax, tspan, sigma_pred)
end

save("untrained_model.png", fig)