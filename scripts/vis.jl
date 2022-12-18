using GLMakie

# scene = Scene()
# surface!(scene, xs, ys, zs, axis = (type=Axis3,))
# save("test.png", scene)
# cyl = GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(0., 0., 0.), Point3f(0., 0., 1.), 1.0f0)
# GLMakie.surface(sol.x, sol.y, sol.u[:, :, 1])
# mesh(cyl, colormap = :veridis)

# scene = Scene(axis = (type = Axis3,))
# # GLMakie.surface!(scene, sol.x, sol.y, sol.u[:, :, 1])
# GLMakie.mesh!(scene, GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(0., 0., 0.), Point3f(0., 0., 1.), 1.0f0))
# GLMakie.mesh!(scene, GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(2., 2., 0.), Point3f(2., 2., 1.), 1.0f0))
# cam3d!(scene)
# scene
# save("surf.png", scene)

fig = Figure()
ax = Axis3(
    fig[1,1],
    title="3D Wave",
    xlabel = "X",
    ylabel = "Y",
    zlabel = "U",
    aspect = :data,
    )

xlims!(ax, -10.0, 10.0)
ylims!(ax, -10.0, 10.0)
zlims!(ax, -1.0, 1.0)


mesh!(
    ax,
    GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(0., 0., 0.), Point3f(0., 0., 2.), 1.0f0),
    color = :blue,
    shading = true
    )

mesh!(
    ax,
    GLMakie.GeometryBasics.Cylinder3{Float32}(Point3f(2., 2., 0.), Point3f(2., 2., 2.), 1.0f0),
    color = :blue,
    shading = true
    )   
fig