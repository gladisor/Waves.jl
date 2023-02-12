export Value, Actor, PPO, reinforcement!, regression!, DistributedPPO, update_actors!, run_episodes!, evaluate, soft_update!, hard_update!, GaussianActor

struct GaussianActor
    base
    μ
    logσ
    a_lim
end

Flux.@functor GaussianActor
Flux.trainable(actor::GaussianActor) = (actor.base, actor.μ, actor.logσ)

function (actor::GaussianActor)(s)
    x = actor.base(s)
    μ = actor.μ(x)
    σ = exp.(clamp.(actor.logσ(x), -20, 10))
    ϵ = randn(Float32, size(σ)) |> gpu
    a = tanh.(μ .+ σ .* ϵ) * actor.a_lim
    return a, μ, σ
end

struct Value
    layers::Chain
end

Flux.@functor Value

function Value(s_size::Int, h_size::Int; σ = tanh)
    layers = Chain(
        Dense(s_size, h_size, σ),
        Dense(h_size, h_size, σ),
        Dense(h_size, 1))

    return Value(layers)
end

function (value::Value)(s)
    return value.layers(s)
end

struct Actor
    pre::Chain
    μ::Dense
    logσ::Dense
    a_lim::Real
end

Flux.@functor Actor
Flux.trainable(actor::Actor) = (actor.pre, actor.μ, actor.logσ)

function Actor(s_size::Int, h_size::Int, a_size::Int; a_lim::Real = 1.0, σ = tanh)
    pre = Chain(
        Dense(s_size, h_size, σ),
        Dense(h_size, h_size, σ)
        )

    μ = Dense(h_size, a_size)
    logσ = Dense(h_size, a_size)

    return Actor(pre, μ, logσ, a_lim)
end

function (actor::Actor)(s)
    x = actor.pre(s)
    μ = actor.μ(x)
    # σ = exp.(clamp.(actor.logσ(x), -20, 10))
    σ = ones(Float32, size(μ)) |> gpu
    ϵ = randn(Float32, size(σ)) |> gpu
    a = tanh.(μ .+ σ .* ϵ) * actor.a_lim
    return a, μ, σ
end

mutable struct PPO <: AbstractPolicy
    actor
    critic
    target_actor
    target_critic

    actor_opt
    critic_opt

    actor_loss
    critic_loss::Float32

    γ::Float32
    ρ::Float32
    ϵ::Float32

    l1_weight
    l2_weight
    l3_weight
end

function PPO(;actor, critic, actor_opt, critic_opt, γ, ρ, ϵ, l1_weight = 1.0f0, l2_weight = 1.0f0, l3_weight = 0.001f0)

    return PPO(
        actor |> gpu, critic |> gpu, 
        deepcopy(actor) |> gpu, deepcopy(critic) |> gpu, 
        actor_opt, critic_opt, 
        0.0f0, 0.0f0, γ, ρ, ϵ, l1_weight, l2_weight, l3_weight)
end

function (ppo::PPO)(env::DesignEnvironment)
    s = Flux.batch([state(env)])
    a, _, _ = ppo.actor(s |> gpu)
    return cpu(vec(a))
end

function soft_update!(target, source, ρ)
    for (targ, src) in zip(Flux.params(target), Flux.params(source))
        targ .= ρ .* targ .+ (1 - ρ) .* src
    end
end

function hard_update!(target, source)
    soft_update!(target, source, 0.0f0)
end

function regression!(ppo::PPO, θ, batch)
    s, a, r, t, s′ = batch

    ∇ = Flux.gradient(θ) do 
        y = r .+ ppo.γ * (1 .- t) .* vec(ppo.target_critic(s′))
        δ = y .- vec(ppo.critic(s))
        loss = mean(δ .^ 2)

        Flux.ignore() do 
            ppo.critic_loss = cpu(loss)
        end

        return loss
    end

    Flux.update!(ppo.critic_opt, θ, ∇)
end

normpdf(μ, σ, x) = 1 ./ (σ * sqrt(2 * pi)) .* exp.(-0.5 * ((x .- μ) ./ σ) .^ 2)

function reinforcement!(ppo::PPO, θ, batch)
    s, a, r, t, s′ = batch
    
    ∇ = Flux.gradient(θ) do
        _, μ_old, σ_old = ppo.target_actor(s)
        old_p_a = prod(normpdf(μ_old, σ_old, a), dims = 1) |> vec

        _, μ, σ = ppo.actor(s)
        p_a = prod(normpdf(μ, σ, a), dims = 1) |> vec
        ratio = p_a ./ old_p_a

        y = r .+ ppo.γ * (1 .- t) .* vec(ppo.target_critic(s′))
        δ = y .- vec(ppo.critic(s))
        δ = (δ .- mean(δ)) ./ std(δ)

        l1 = -mean(min.(ratio .* δ, clamp.(ratio, 1.0f0 - ppo.ϵ, 1.0f0 + ppo.ϵ) .* δ)) * ppo.l1_weight
        l2 = mean((1 .- ratio) .^ 2) * ppo.l2_weight
        l3 = mean(ratio .* log.(p_a)) * ppo.l3_weight

        loss =  l1 + l2 + l3

        Flux.ignore() do 
            ppo.actor_loss = [cpu(l1), cpu(l2), cpu(l3)]
        end

        return loss
    end

    Flux.update!(ppo.actor_opt, θ, ∇)
end

mutable struct PPOActor <: AbstractPolicy
    actor
end

function (actor::PPOActor)(env::DesignEnvironment)
    a, _, _ = actor.actor(gpu(Flux.batch([state(env)])))
    return cpu(vec(a))
end

mutable struct DistributedPPO
    learner::PPO
    actors::DArray{<:PPOActor}
end

function DistributedPPO(;kwargs...)
    @assert nworkers() > 1 "Needs more than one worker."

    learner = PPO(;kwargs...)
    actors = dfill(PPOActor(cpu(learner.actor)), nworkers())

    if CUDA.has_cuda_gpu()
        n_gpu = length(Flux.devices())

        @sync @distributed for w in workers() ## distribute workers across gpus
            Flux.device!(w % n_gpu)
            actors[:L][1].actor = gpu(actors[:L][1].actor)
        end
    end

    return DistributedPPO(learner, actors)
end

function update_actors!(dppo::DistributedPPO)
    n_gpu = CUDA.has_cuda_gpu() ? length(Flux.devices()) : 0

    @sync @distributed for w in workers() ## copy parameters from main network

        if CUDA.has_cuda_gpu()
            Flux.device!(w % n_gpu)
        end

        Flux.loadparams!(dppo.actors[:L][1].actor, Flux.params(dppo.learner.actor))
    end

    return nothing
end

function run_episodes!(dppo::DistributedPPO, envs::DArray{ <: DesignEnvironment}, data::DArray{<: CircularArraySARTTrajectory}, episodes::Int)

    n_gpu = CUDA.has_cuda_gpu() ? length(Flux.devices()) : 0

    @time @sync @distributed for w ∈ workers()

        if CUDA.has_cuda_gpu()
            Flux.device!(w % n_gpu)
        end

        agent = Agent(policy = dppo.actors[:L][1], trajectory = data[:L][1])
        run(agent, envs[:L][1], StopAfterEpisode(episodes))
    end

    return nothing
end

function generate_sarts(ppo, envs, episodes)::Tuple
    data = distribute([build_buffer(envs[i], episodes) for (i, _) ∈ enumerate(workers())])
    run_episodes!(ppo, envs, data, episodes)
    data = convert(Vector, data) ## send the distributed array back to main process
    s, a, r, t, s′ = sarts(data)
    return (s, a, r, t, s′)
end

function evaluate(dppo::DistributedPPO, envs::DArray{ <: DesignEnvironment}; path = nothing)
    n_gpu = CUDA.has_cuda_gpu() ? length(Flux.devices()) : 0
    episode_rewards = dones(nworkers())

    @sync @distributed for w ∈ workers()

        if CUDA.has_cuda_gpu()
            Flux.device!(w % n_gpu)
        end

        trpe = TotalRewardPerEpisode(is_display_on_exit = false)
        sd = SaveData(envs[:L][1])
        hook = ComposedHook(trpe, sd)

        run(dppo.actors[:L][1], envs[:L][1], StopWhenDone(), hook)
        episode_rewards[:L][1] = sum(trpe.rewards)

        if !isnothing(path)
            render!(sd.data, path = joinpath(path, "episode_worker=$(w).mp4"))
        end
    end

    episode_rewards = convert(Vector, episode_rewards)

    return mean(episode_rewards)
end