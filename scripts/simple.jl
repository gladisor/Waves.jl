using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using IfElse

using GLMakie

x_min = y_min = -10.0
x_max = y_max = 10.0
t_max = 1.0

@parameters x [bounds = (x_min, x_max)]
@parameters y [bounds = (y_min, y_max)]
@parameters t [bounds = (0.0, t_max)]
@variables u(..)

Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)
Dxx = Differential(x) ^ 2
Dyy = Differential(y) ^ 2
Dtt = Differential(t) ^ 2

M = 2
@parameters x_pos[1:M], y_pos[1:M]
@parameters radii[1:M]
@parameters wavespeed

function C(x, y, t)
    c = wavespeed
    
    for i in 1:M
        c = IfElse.ifelse((x - x_pos[i]) ^ 2 + (y - y_pos[i]) ^ 2 < radii[i] ^ 2, 0.0, c)
    end

    return c
end

eq = Dtt(u(x, y, t)) ~ C(x, y, t) ^ 2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))

bcs = [
    u(x_min, y, t) ~ 0., 
    u(x_max, y, t) ~ 0.,
    u(x, y_min, t) ~ 0.,
    u(x, y_max, t) ~ 0.,
    Dx(u(x_min, y, t)) ~ 0., 
    Dx(u(x_max, y, t)) ~ 0.,
    Dy(u(x, y_min, t)) ~ 0.,
    Dy(u(x, y_max, t)) ~ 0.,
    Dt(u(x, y, 0.0)) ~ 0.,
    u(x, y, 0.0) ~ exp(-1 * (x^2 + y^2))
]

domain = [
        x ∈ getbounds(x),
        y ∈ getbounds(y),
        t ∈ getbounds(t)
        ]

x_pos_params = [-3.0, -3.0]
y_pos_params = [-2.0, 2.0]
radii_params = [1.0, 1.0]

@named sys = PDESystem(
    eq, 
    bcs, 
    domain, 
    [x, y, t], 
    [u(x, y, t)], 
    [
        wavespeed => 4.0,
        collect(x_pos .=> x_pos_params)...,
        collect(y_pos .=> y_pos_params)...,
        collect(radii .=> radii_params)...
    ]
    )

n = 30
disc = MOLFiniteDifference([x => n, y => n], t)
prob = discretize(sys, disc)
prob = remake(
    prob, 
    p = [1.0, -5.0, -5.0, -2.0, 2.0, 1.0, 1.0], 
    tspan = (0.0, 10.0))

grid = get_discrete(sys, disc)

iter = init(prob, Tsit5())

for i ∈ 1:100
    display(iter.t)
    step!(iter, 0.01, true)
end

u_sol = iter.sol[grid[u(x, y, t)]]
fig = Figure(resolution = (1920, 1080), fontsize = 20)

ax = Axis3(
    fig[1,1],
    aspect = (1, 1, 1),
    perspectiveness = 0.5,
    title="3D Wave",
    xlabel = "X",
    ylabel = "Y",
    zlabel = "Z",
    )

xlims!(ax, getbounds(x)...)
ylims!(ax, getbounds(y)...)
zlims!(ax, 0.0, 1.0)

x_domain = grid[x]
y_domain = grid[y]

record(fig, "iterator.mp4", axes(u_sol, 1)) do i
    GLMakie.empty!(ax.scene)
    GLMakie.surface!(ax, x_domain, y_domain, u_sol[i], colormap = :ice, shading = false)
end