include("dependencies.jl")

build_cost(model::WaveMPC, s::ScatteredWaveEnvState, a::AbstractDesign) = mean(model(s, a))

function optimize_action(model::WaveMPC, s::ScatteredWaveEnvState, a::AbstractDesign)
    opt_state = Optimisers.setup(Optimisers.Adam(0.1),  a)

    for i in 1:5
        z_wave = model.wave_encoder(s.wave_total)

        cost, back = pullback(_a -> build_cost(model, s, _a) + norm(vec(_a)), a)
        gs = back(one(cost))[1]
        opt_state, a = Optimisers.update(opt_state, a, gs)
    end

    return a
end

struct MPCPolicy <: AbstractPolicy
    random::RandomDesignPolicy
    model::WaveMPC
end

function (policy::MPCPolicy)(env::ScatteredWaveEnv)
    s = gpu(state(env))
    a = gpu(policy.random(env))
    a = optimize_action(policy.model, s, a)
    return a
end

function (model::WaveMPC)(z_wave::AbstractMatrix{Float32}, da) #design::AbstractDesign, action::AbstractDesign)
    design, action = da
    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    zi = hcat(z_wave, z_design)
    z = model.iter(zi)
    z_wave = z[:, [1, 2], end]
    zf = z[:, :, end]
    return z_wave, zf
end

function train(model::WaveMPC, train_loader::DataLoader, epochs::Int)
    opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)

    for i in 1:epochs
        train_loss = 0.0f0

        for (s, a, σ) in train_loader
            s, a, σ = gpu(s[1]), gpu(a[1]), gpu(σ[1])

            loss, back = pullback(_model -> build_loss(model, s, a, σ), model)
            gs = back(one(loss))[1]

            opt_state, model = Optimisers.update(opt_state, model, gs)
            train_loss += loss
        end

        print("Epoch: $i, Loss: ")
        println(train_loss / length(train_loader))
    end

    return model
end

path = "results/radii/WaveMPC"
env = BSON.load(joinpath(path, "env.bson"))[:env] |> gpu
policy = RandomDesignPolicy(action_space(env))
;

reset!(env)
s = state(env)
a = gpu(policy(env))

design = s.design
design_size = length(vec(design))

h_size = 128
latent_elements = 512
latent_dim = OneDim(grid_size, latent_elements)
latent_dynamics = ForceLatentDynamics(ambient_speed, build_gradient(latent_dim), dirichlet(latent_dim))

wave_encoder = Chain(
    WaveEncoder(6, 8, 2, tanh), 
    Dense(1024, latent_elements, tanh),
    z -> hcat(z[:, 1], z[:, 2] * 0.0f0))

design_encoder = Chain(
    Dense(2 * design_size, h_size, relu), 
    Dense(h_size, 2 * latent_elements),
    z -> reshape(z, latent_elements, :),
    z -> hcat(tanh.(z[:, 1]), sigmoid.(z[:, 2]))
    )

iter = Integrator(runge_kutta, latent_dynamics, ti, dt, steps)
mlp = Chain(flatten, Dense(latent_elements * 4, h_size, relu), Dense(h_size, 1), vec)

model = gpu(WaveMPC(wave_encoder, design_encoder, iter, mlp))
data = generate_episode_data(policy, env, 10)

train_loader = Flux.DataLoader(prepare_data(data, 0), shuffle = true)
model = train(model, train_loader, 2)
;