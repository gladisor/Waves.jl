using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using IfElse

using GLMakie

struct Cylinder
    x
    y
    r
    c
end

function Cylinder(x, y)
    return Cylinder(x, y, 1.0, 0.0)
end

function Base.:(:)(initial::Cylinder, n::Int, final::Cylinder)
    xs = range(initial.x, final.x, length = n)
    ys = range(initial.y, final.y, length = n)
    rs = range(initial.r, final.r, length = n)
    cs = range(initial.c, final.c, length = n)
    return Cylinder.(xs, ys, rs, cs)
end

function GLMakie.mesh!(ax::Axis3, cyl::Cylinder)
    GLMakie.mesh!(ax, GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(cyl.x, cyl.y, 0.), Point3f(cyl.x, cyl.y, 1.0), cyl.r), color = :grey)
end

function design_parameters(cyl::Cylinder)
    return [cyl.x, cyl.y, cyl.r, cyl.c]
end

function Base.in(point::Tuple, cyl::Cylinder)
    x, y = point
    return (x - cyl.x) ^ 2 + (y - cyl.y) ^ 2 < cyl.r ^ 2
end

struct PositionAction
    dx::Real
    dy::Real
end

function Base.:+(cyl::Cylinder, a::PositionAction)
    return Cylinder(cyl.x + a.dx, cyl.y + a.dy, cyl.r, cyl.c)
end

struct RadiusAction
    dr::Real
end

function Base.:+(cyl::Cylinder, a::RadiusAction)
    return Cylinder(cyl.x, cyl.y, cyl.r + a.dr, cyl.c)
end

struct WaveSpeedAction
    dc::Real
end

function Base.:+(cyl::Cylinder, a::WaveSpeedAction)
    return Cylinder(cyl.x, cyl.y, cyl.r, cyl.c + a.dc)
end

function design_parameters(design::Vector{Cylinder})::Matrix
    return hcat(design_parameters.(design)...)'
end

struct DesignParameters
    initial
    final
end

function DesignParameters(s::Tuple{Int, Int})
    @parameters initial[(1:i for i ∈ s)...], final[(1:i for i ∈ s)...]
    return DesignParameters(initial, final)
end

function DesignParameters(design::Vector{Cylinder})
    ps = design_parameters(design)
    return DesignParameters(size(ps))
end

function interpolate(dp::DesignParameters, param::Int, t)
    initial_param = collect(dp.initial[:, param])
    final_param = collect(dp.final[:, param])
    return initial_param .+ (final_param .- initial_param) .* t
end

initial_design = Vector{Cylinder}([
    Cylinder(-5.0, -6.0),
    Cylinder(-5.0, -3.0),
    Cylinder(-5.0, 0.0),
    Cylinder(-5.0, 3.0),
    Cylinder(-5.0, 6.0),
    ])

dp = DesignParameters(initial_design)
M = length(initial_design)

# a = [3]
# a = [RadiusAction(a[i]) for i ∈ 1:M]
a = [[10.0, 10.0, 10.0, 10.0, 10.0] [0.0, 0.0, 0.0, 0.0, 0.0]]
a = [PositionAction(a[i, :]...) for i ∈ 1:M]

final_design = initial_design .+ a
x_min = y_min = -10.0
x_max = y_max = 10.0
t_max = 5.0

@parameters x [bounds = (x_min, x_max)]
@parameters y [bounds = (y_min, y_max)]
@parameters t [bounds = (0.0, t_max)]
@variables u(..)

@parameters T[1:2]
@parameters wavespeed

Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)
Dxx = Differential(x) ^ 2
Dyy = Differential(y) ^ 2
Dtt = Differential(t) ^ 2

function C(x, y, t)
    c = wavespeed

    t_norm = (t - T[1]) / (T[2] - T[1])
    x_interp = interpolate(dp, 1, t_norm)
    y_interp = interpolate(dp, 2, t_norm)
    r_interp = interpolate(dp, 3, t_norm)
    c_interp = interpolate(dp, 4, t_norm)

    cylinders = Cylinder.(x_interp, y_interp, r_interp, c_interp)

    for i in 1:M
        c = IfElse.ifelse((x, y) ∈ cylinders[i], cylinders[i].c, c)
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

@named sys = PDESystem(
    eq, 
    bcs, 
    domain, 
    [x, y, t], 
    [u(x, y, t)], 
    [
        wavespeed => 2.0,
        collect(T .=> [0.0, t_max])...,
        collect(dp.initial .=> design_parameters(initial_design))...,
        collect(dp.final .=> design_parameters(final_design))...,
    ])

n = 30
disc = MOLFiniteDifference([x => n, y => n], t)
prob = discretize(sys, disc)

prob = remake(
    prob, 
    p = [
        2.0, # ambient wave speed
        0.0, t_max,
        design_parameters(initial_design)...,
        design_parameters(final_design)...
        ], 
    )

grid = get_discrete(sys, disc)
iter = init(prob, Tsit5(); tstops = collect(0.0:0.05:t_max), advance_to_tstop = true)

reinit!(iter)
solve!(iter)

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
zlims!(ax, 0.0, 5.0)

x_domain = grid[x]
y_domain = grid[y]

timesteps = size(u_sol, 1)

interpolated_design = collect(zip([initial_design[i]:timesteps:final_design[i] for i ∈ 1:M]...))

record(fig, "parameters.mp4", axes(u_sol, 1)) do i
    empty!(ax.scene)

    for cyl ∈ interpolated_design[i]
        mesh!(ax, cyl)
    end

    surface!(ax, x_domain, y_domain, u_sol[i], colormap = :ice, shading = false)
end