export render!

const FRAMES_PER_SECOND = 24

function render!(policy::AbstractPolicy, env::WaveEnv, seconds::Float32 = env.actions * 0.5f0)
    tspans = []
    interps = DesignInterpolator[]
    u_tots = []
    # u_incs = []

    reset!(env)
    while !is_terminated(env)
        tspan, interp, u_tot, u_inc = cpu(env(policy(env)))
        push!(tspans, tspan)
        push!(interps, interp)
        push!(u_tots, u_tot)
        # push!(u_incs, u_inc)
        println(env.time_step)
    end

    tspan = flatten_repeated_last_dim(hcat(tspans...))

    u_tot = flatten_repeated_last_dim(cat(u_tots..., dims = 4))
    u_tot = linear_interpolation(tspan, Flux.unbatch(u_tot))

    # u_inc = flatten_repeated_last_dim(cat(u_incs..., dims = 4))
    # u_inc = linear_interpolation(tspan, Flux.unbatch(u_inc))

    frames = Int(round(FRAMES_PER_SECOND * seconds))
    tspan = range(tspan[1], tspan[end], frames)

    fig = Figure()
    ax1 = Axis(fig[1, 1], aspect = 1.0f0)
    # ax2 = Axis(fig[1, 2], aspect = 1.0f0)

    dim = cpu(env.dim)
    @time record(fig, "vid.mp4", tspan, framerate = FRAMES_PER_SECOND) do t
        empty!(ax1)
        # empty!(ax2)

        heatmap!(ax1, dim.x, dim.y, u_tot(t), colormap = :ice, colorrange = (-1.0f0, 1.0f0))
        mesh!(ax1, multi_design_interpolation(interps, t))
        # heatmap!(ax2, dim.x, dim.y, u_inc(t), colormap = :ice, colorrange = (-1.0f0, 1.0f0))
    end
    return nothing
end