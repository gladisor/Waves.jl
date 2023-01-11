
action = Cylinder(1.0, -1.0, 0.0, 0.0)
action2 = Cylinder(0.0, 3.0, 0.0, 0.0)

actions = [
    Cylinder(1.0, -1.0, 0.0, 0.0),
    Cylinder(0.0, 2.0, 0.0, 0.0),
    Cylinder(1.0, 1.0, 0.0, 0.0)]

ps = [
    wave.speed => 2.0,
    (design_parameters(pd.initial) .=> design_parameters(pd.design))...,
    (design_parameters(pd.final) .=> design_parameters(new_design))...,
    pd.t_initial => t0,
    pd.t_final => t0 + dt
    ]

eq = Waves.wave_equation(wave, pd)

bcs = [
    wave.u(dims(wave)..., 0.0) ~ exp(-1.0 * (wave.dim.x ^ 2 + wave.dim.y ^ 2)),
    Waves.boundary_conditions(wave)...
    ]

@named sys = PDESystem(
    eq, 
    bcs, 
    Waves.get_domain(wave, t_max = tf), 
    [dims(wave)..., wave.t], 
    [wave.u(dims(wave)..., wave.t)], 
    ps)

n = 30
disc = MOLFiniteDifference([Pair.(Waves.dims(wave), n)...], wave.t)
prob = discretize(sys, disc)
grid = get_discrete(sys, disc)
iter = init(prob, Tsit5(), advance_to_tstop = true, saveat = 0.05)

reinit!(iter)
add_tstop!(iter, iter.t + dt)
step!(iter)
cyls = range(pd.design, new_design, length(iter.sol))
pd.design = new_design
new_design = pd.design + action2
iter.p[2:5] .= design_parameters(pd.design)
iter.p[6:9] .= design_parameters(new_design)
iter.p[end-1] = iter.p[end]
iter.p[end] = iter.t + dt
add_tstop!(iter, iter.t + dt)
step!(iter)
cyls = vcat(cyls, range(pd.design, new_design, length(iter.sol) - length(cyls)))
sol = iter.sol[grid[wave.u(Waves.dims(wave)..., wave.t)]]

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

xlims!(ax, getbounds(wave.dim.x)...)
ylims!(ax, getbounds(wave.dim.y)...)
zlims!(ax, 0.0, 5.0)

record(fig, "animations/3d.gif", axes(sol, 1)) do i
    empty!(ax.scene)
    surface!(
        ax, 
        collect(grid[wave.dim.x]),
        collect(grid[wave.dim.y]),
        sol[i], 
        colormap = :ice, 
        shading = false)

    mesh!(ax, cyls[i])
end