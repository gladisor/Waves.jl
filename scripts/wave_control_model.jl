function optimize_action(opt_state::NamedTuple, model::AbstractWaveControlModel, s::ScatteredWaveEnvState, a::AbstractDesign, steps::Int)
    println("optimize_action")

    for i in 1:steps
        cost, back = pullback(_a -> sum(model(s, _a)), a)
        gs = back(one(cost))[1]
        opt_state, a = Optimisers.update(opt_state, a, gs)
        println(cost)
    end

    return a
end

struct WaveControlModel <: AbstractWaveControlModel
    wave_encoder::Union{WaveEncoder, Chain}
    design_encoder::Union{DesignEncoder, Chain}
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveControlModel

function encode(model::WaveControlModel, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    u = model.wave_encoder(wave)
    v = u * 0.0f0
    return hcat(u, v, model.design_encoder(design, action))
end

function (model::WaveControlModel)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    zi = encode(model, wave, design, action)
    z = model.iter(zi)
    return model.mlp(z)
end

function (model::WaveControlModel)(s::ScatteredWaveEnvState, a::AbstractDesign)
    return model(s.wave_total, s.design, a)
end

function build_control_sequence(action::AbstractDesign, steps::Int)
    return [zero(action) for i in 1:steps]
end

function build_mpc_cost(model::WaveControlModel, s::ScatteredWaveEnvState, control_sequence::Vector{ <: AbstractDesign})
    cost = 0.0f0

    d1 = s.design
    c1 = model.design_encoder(d1, control_sequence[1])
    z1 = hcat(model.wave_encoder(s.wave_total), c1)
    z = model.iter(z1)
    cost = cost + sum(model.mlp(z))

    d2 = d1 + control_sequence[1]
    c2 = model.design_encoder(d2, control_sequence[2])
    z2 = hcat(z[:, 1:2, end], c2)
    z = model.iter(z2)
    cost = cost + sum(model.mlp(z))

    d3 = d2 + control_sequence[2]
    c3 = model.design_encoder(d3, control_sequence[3])
    z3 = hcat(z[:, 1:2, end], c3)
    z = model.iter(z3)
    cost = cost + sum(model.mlp(z))

    return cost
end

struct EpisodeData
    states::Vector{ScatteredWaveEnvState}
    actions::Vector{<: AbstractDesign}
    tspans::Vector{Vector{Float32}}
    sigmas::Vector{Vector{Float32}}
end

Flux.@functor EpisodeData

function generate_episode_data(policy::AbstractPolicy, env::ScatteredWaveEnv)
    states = ScatteredWaveEnvState[]
    actions = AbstractDesign[]
    tspans = []
    sigmas = []

    reset!(env)

    while !is_terminated(env)
        action = gpu(policy(env))
        tspan = build_tspan(time(env), env.dt, env.integration_steps)

        push!(states, cpu(state(env)))
        push!(actions, cpu(action))
        push!(tspans, tspan)

        @time env(action)

        push!(sigmas, cpu(env.σ))
    end

    return EpisodeData(states, actions, tspans, sigmas)
end

function generate_episode_data(policy::AbstractPolicy, env::ScatteredWaveEnv, episodes::Int)
    data = EpisodeData[]

    for episode in 1:episodes
        episode_data = generate_episode_data(policy, env)
        push!(data, episode_data)
    end

    return data
end

function train(model::AbstractWaveControlModel, train_loader::Flux.DataLoader, epochs::Int)
    opt_state = Optimisers.setup(Optimisers.Adam(1e-4), model)

    for i in 1:epochs
        train_loss = 0.0f0

        for batch in train_loader
            s, a, sigma = batch
            s, a, sigma = gpu(s[1]), gpu(a[1]), gpu(sigma[1])

            loss, back = pullback(_model -> mse(_model(s, a), sigma), model)
            gs = back(one(loss))[1]

            opt_state, model = Optimisers.update(opt_state, model, gs)
            train_loss += loss
        end

        print("Epoch: $i, Loss: ")
        println(train_loss / length(data))
    end

    return model
end

struct PercentageWaveControlModel <: AbstractWaveControlModel
    wave_encoder::WaveEncoder
    wave_encoder_mlp::Chain

    design_encoder::DesignEncoder
    design_encoder_mlp::Chain

    iter::Integrator
    mlp::Chain
end

Flux.@functor PercentageWaveControlModel

function encode(model::PercentageWaveControlModel, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    u = vec(model.wave_encoder_mlp(model.wave_encoder(wave)))
    v = u * 0.0f0
    c = model.design_encoder_mlp(model.design_encoder(design, action))
    # v = c * 0.01f0
    return hcat(u, v, c)
end

function (model::PercentageWaveControlModel)(s::ScatteredWaveEnvState, a::AbstractDesign)
    zi = encode(model, s.wave_total, s.design, a)
    z = model.iter(zi)
    return model.mlp(z)
end


struct WaveMPC <: AbstractWaveControlModel
    wave_encoder::Chain
    design_encoder::Chain
    iter::Integrator
    mlp::Chain
end

Flux.@functor WaveMPC

function encode(model::WaveMPC, wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    z_wave = model.wave_encoder(wave)
    z_u = z_wave[:, 1]
    z_v = z_wave[:, 2] * 0.0f0

    z_design = model.design_encoder(vcat(vec(design), vec(action)))
    z_design = reshape(z_design, size(z_design, 1) ÷ 2, 2)
    z_f = tanh.(z_design[:, 1])
    z_c = sigmoid.(z_design[:, 2])
    zi = hcat(z_u, z_v, z_f, z_c)
    return zi
end

function (model::WaveMPC)(wave::AbstractArray{Float32, 3}, design::AbstractDesign, action::AbstractDesign)
    z = model.iter(encode(model, wave, design, action))
    return model.mlp(z)
end

function (model::WaveMPC)(s::ScatteredWaveEnvState, action::AbstractDesign)
    return model(s.wave_total, s.design, action)
end