using Flux

x = Float32[-4.0f0]
ps = Flux.params(x)

function square(x)
    return x .^ 2
end

opt = Descent(0.01f0)

for i ∈ 1:100
        
    gs = Flux.gradient(ps) do 
        sum(square(x))
    end

    Flux.Optimise.update!(opt, ps, gs)
    println(x)
end

