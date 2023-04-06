using Flux
using Flux.Losses: mse

y = exp.(range(0.0, 1.0, 20))

x = randn(10, 20)
m = Chain(Dense(10, 1), vec)

gs = gradient(_x -> mse(y, m(_x)), x)[1]

display(mse(y, m(x)))
display(mse(y, m(x .+ gs)))