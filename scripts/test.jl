include("dependencies.jl")

struct SingleImageInput end
Flux.@functor SingleImageInput
(input::SingleImageInput)(img::AbstractArray{Float32, 3}) = img[:, :, :, :]

function build_hypernet_control_model(;
        h_size::Int, 
        act::Function, 
        latent_dim::OneDim, 
        ambient_speed::Float32,
        design_action_size::Int,
        dt::Float32,
        steps::Int
        )

    wave_embedder = Chain(
        Dense(1, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, 2),
        LatentWaveActivation(ambient_speed)
    )

    wave_embedder_ps, wave_embedder_restructure = destructure(wave_embedder)

    wave_embedder_restructure(wave_embedder_ps)

    wave_encoder = Chain(
        SingleImageInput(),
        MeanPool((2, 2)), ## Initial dimentionality reduction
        DownBlock(2, 1, 32, act),
        InstanceNorm(32),
        DownBlock(2, 32, 32, act),
        InstanceNorm(32),
        DownBlock(2, 32, 64, act),
        InstanceNorm(64),
        DownBlock(2, 64, 128, act),
        GlobalMaxPool(),
        flatten,
        Dense(128, 256, act),
        Dense(256, length(wave_embedder_ps), bias = false),
        vec,
        wave_embedder_restructure,
        Field(latent_dim))

    design_embedder = Chain(
        Dense(1, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, h_size, act),
        Dense(h_size, 2),
        LatentDesignActivation()
    )

    design_embedder_ps, design_embedder_restructure = destructure(design_embedder)

    design_encoder = Chain(
        Dense(design_action_size, h_size, act),
        LayerNorm(h_size),
        Dense(h_size, h_size, act),
        LayerNorm(h_size),
        Dense(h_size, length(design_embedder_ps), bias = false),
        vec,
        design_embedder_restructure,
        Field(latent_dim)
    )

    grad = build_gradient(latent_dim)
    pml = build_pml(latent_dim, 2.0f0, 1.0f0)
    bc = dirichlet(latent_dim)
    dynamics = ForceLatentDynamics(AIR, 5000.0f0, grad, pml, bc)
    iter = Integrator(runge_kutta, dynamics, 0.0f0, dt, steps)

    mlp = Chain(
        flatten,
        Dense(4 * size(latent_dim, 1), h_size, gelu),
        Dense(h_size, h_size, gelu),
        Dense(h_size, 1),
        vec
    )

    return WaveControlModel(wave_encoder, design_encoder, iter, mlp)
end

function visualize_latent_wave!(latent_dim::OneDim, model::WaveControlModel, s::WaveEnvState, actions::Vector{<: AbstractDesign}, tspan::AbstractMatrix; path::String)

    tspan = cpu(tspan)
    tspan = vcat(tspan[1], vec(tspan[2:end, :]))

    z_wave = model.wave_encoder(s.wave_total)
    h = (z_wave, s.design)

    zs = []

    for (i, a) in enumerate(actions)
        z_wave, design = h
        z_design = model.design_encoder(vcat(vec(design), vec(a)))
        z = model.iter(hcat(z_wave, z_design))
        h = (z[:, [1, 2], end], design + a)

        if i == 1
            push!(zs, z)
        else
            push!(zs, z[:, :, 2:end])
        end
    end

    z = cat(zs..., dims = ndims(zs[1]))

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0f0)
    heatmap!(ax1, latent_dim.x, tspan, cpu(z[:, 1, :]), colormap = :ice)

    ax2 = Axis(fig[1, 2], aspect = 1.0f0)
    heatmap!(ax2, latent_dim.x, tspan, cpu(z[:, 2, :]), colormap = :ice)

    ax3 = Axis(fig[2, 1], aspect = 1.0f0)
    heatmap!(ax3, latent_dim.x, tspan, cpu(z[:, 3, :]), colormap = :ice)

    ax4 = Axis(fig[2, 2], aspect = 1.0f0)
    heatmap!(ax4, latent_dim.x, tspan, cpu(z[:, 4, :]), colormap = :ice)
    save(path, fig)
end


Flux.device!(1)

data_path = "data/hexagon_large_grid"
env = BSON.load(joinpath(data_path, "env.bson"))[:env]
reset!(env)

data = EpisodeData(path = joinpath(data_path, "episode1/episode.bson"))
s = data.states[end]
a = data.actions[end]

design_action = vcat(vec(s.design), vec(a))

latent_elements = 256
latent_dim = OneDim(s.dim.x[end], latent_elements)

model = build_hypernet_control_model(
    h_size = 256, 
    act = gelu, 
    latent_dim = latent_dim,
    ambient_speed = AIR,
    design_action_size = length(design_action),
    dt = env.dt,
    steps = env.integration_steps
    ) |> gpu

states, actions, tspans, sigmas = prepare_data(data, 3)

idx = 10

s = gpu(states[idx])
a = gpu(actions[idx])
t = tspans[idx]
sigma = gpu(sigmas[idx])

z = visualize_latent_wave!(latent_dim, model, s, a, t, path = "latent.png")

# render!(latent_dim, cpu(z), path = "vid.mp4")

# z_wave = model(design_action)
# # z_wave = model(s.wave_total)

# fig = Figure()
# ax1 = Axis(fig[1, 1], aspect = 1.0f0)
# lines!(ax1, latent_dim.x, z_wave[:, 1])

# ax2 = Axis(fig[1, 2], aspect = 1.0f0)
# lines!(ax2, latent_dim.x, z_wave[:, 2])

# save("z_wave.png", fig)













# model = Chain(
#     SingleImageInput(),
#     MeanPool((2, 2)),
# )

# sc = s.wave_total .- s.wave_incident

# fig = Figure()
# ax1 = Axis(fig[1, 1], aspect = 1.0f0)
# heatmap!(ax1, model(s.wave_total)[:, :, 1], colormap = :ice)

# ax2 = Axis(fig[1, 2], aspect = 1.0f0)
# heatmap!(ax2, model(sc)[:, :, 1], colormap = :ice)
# save("wave.png", fig)