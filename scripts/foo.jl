using Flux
using Flux: params
using ChainRulesCore

struct Foo
    A::Matrix
    c::Float64
end

function foo_mul(foo::Foo, b::AbstractArray)
    return foo.A * b
end

function ChainRulesCore.rrule(::typeof(foo_mul), foo::Foo, b::AbstractArray)
    println("calling foo rrule")
    y = foo_mul(foo, b)

    function foo_mul_pullback(ȳ)

        f̄ = NoTangent()
        f̄oo = Tangent{Foo}(; A=ȳ * b', c=ZeroTangent())
        b̄ = @thunk(foo.A' * ȳ)

        return f̄, f̄oo, b̄
    end

    return y, foo_mul_pullback
end

foo = Foo(randn(2, 2), 1.0)
b = randn(2)

gs = gradient(foo -> sum(foo_mul(foo, b)), foo)[1]
opt = Descent(0.01)
Flux.Optimise.update!(opt, foo, gs)