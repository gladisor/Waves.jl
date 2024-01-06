using NeuralPDE, Lux, CUDA, Random, ComponentArrays
using Optimization
using OptimizationOptimisers
import ModelingToolkit: Interval

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min = 0.0
t_max = 2.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0

# 2D PDE
eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
# Initial and boundary conditions
bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
    u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
    u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
    u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
    u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)]

# Neural network
inner = 25
chain = Chain(Dense(3, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, inner, Lux.σ),
              Dense(inner, 1))

strategy = GridTraining(0.05)
ps = Lux.setup(Random.default_rng(), chain)[1]
ps = ps |> ComponentArray |> gpu .|> Float64
discretization = PhysicsInformedNN(chain,
                                   strategy,
                                   init_params = ps)

@named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
prob = discretize(pde_system, discretization)
symprob = symbolic_discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, Adam(0.01); callback = callback, maxiters = 2500)