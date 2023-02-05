using Flux

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
    ϵ = randn(Float32, size(σ))
    a = tanh.(μ .+ σ .* ϵ) * actor.a_lim
    return a, μ, σ
end