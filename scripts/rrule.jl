using Flux
using ChainRulesCore
using Optimisers

struct Linear
    W::Matrix
    b::Vector
end

function (l::Linear)(x::Vector)
    return l.W * x .+ l.b
end

Flux.@functor Linear

function ChainRulesCore.rrule(l::Linear, x::Vector)
    println("calling linear rrule")

    y = l(x)

    function linear_back(Δ)
        println("calling linear back")

        dW = Δ * x'
        db = Δ
        dx = (Δ' * l.W)'

        tangent = Tangent{Linear}(;W = dW, b = db)
        return tangent, dx
    end

    return y, linear_back
end

struct TestLayer
    l::Linear
end

Flux.@functor TestLayer

function (layer::TestLayer)(x::Vector)
    return layer.l(x)
end

# layer = TestLayer(Linear(randn(2, 2), zeros(2)))

model = Flux.Chain(
    # layer
    Linear(randn(2, 2), zeros(2)),
    )

x = randn(2)

gs, _ = gradient((_model, _x) -> sum(_model(_x)), model, x)


state_tree = Optimisers.setup(Optimisers.Adam(1.0f0), model)


display(model)
state_tree, model = Optimisers.update(state_tree, model, gs)
display(model)