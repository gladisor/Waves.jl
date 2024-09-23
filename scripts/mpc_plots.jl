using Waves, Flux, CairoMakie, BSON, Statistics

colors = [  :red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown, :pink,
                :lime, :teal, :violet, :gold, :indigo, :olive, :navy, :coral, :turquoise, :salmon ]

function get_best_gbo(gbo_data, signal_index)
    steady_state_values = []
    for i in 1:length(gbo_data)
        for j in 1:length(gbo_data[i])
            push!(steady_state_values, gbo_data[i][j][:signal][signal_index, end])
        end
    end
    best_value = signal_index == 3 ? minimum(steady_state_values) : maximum(steady_state_values)
    return ones(Float32, length(gbo_data[1][1][:signal][signal_index, :])) * best_value
end

function get_best_gbo_single(gbo_data::Vector{Dict{Symbol, Any}}, signal_index)
    return get_best_gbo([gbo_data], signal_index)
end

function multiple_mpc_rendering(data_array, data_description, title, output_path, signal_index, gbo_ss=nothing)
    fig = Figure(;size = (2400, 1800))

    max_bound = 0
    max_energy = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        max_energy = max(max_energy, maximum(vcat(vecs...)))
        upper_bound = maximum(mean(vecs) .+ 0.5 * sqrt.(var(vecs)))
        max_bound = max(max_bound, upper_bound)
    end

    ax_master_array = []
    ax_last_array = []
    average_array = []
    upper_bound_array = []
    lower_bound_array = []
    ax_average_array = []

    for i in 1:length(data_array) 
        ax_arr = []
        for j in 1:length(data_array[i])
            push!(ax_arr, Axis(fig[j, 2*i-1], aspect = 1.0, title = "$(String(colors[j]))", xlabel = "Space (m)", ylabel = "Space (m)"))
        end
        ax_last = Axis(fig[1:div(length(data_array[i]), 2)  , 2*i], title = "$title ($(data_description[i]))", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_last, t[1], round(t[end], digits=1))
        ylims!(ax_last, 0.0, max_energy * 1.20)
        
        push!(ax_master_array, ax_arr)
        push!(ax_last_array, ax_last)

        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        push!(average_array, mean(vecs))
        push!(upper_bound_array, average_array[end] .+ sqrt.(var(vecs)))
        push!(lower_bound_array, average_array[end] .- sqrt.(var(vecs)))

        ax_average = Axis(fig[(div(length(data_array[i]), 2) + 1):length(data_array[i]), 2*i], title = "Mean ± 0.5 * Standard Deviation", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_average, t[1], round(t[end], digits=1))
        ylims!(ax_average, 0.0, max_bound * 1.20)
        push!(ax_average_array, ax_average)
    end

    # ax_average = Axis(fig[(length(data_array[1]) + 1):2*length(data_array[1]), 1:(2*length(data_array))], title = "Mean ± 0.5 * Standard Deviation", xlabel = "Time (s)", ylabel = "Energy")
    # xlims!(ax_average, t[1], round(t[end], digits=1))
    # ylims!(ax_average, 0.0, max_bound * 1.20)

    # Bounding box coordinates
    x1, y1, x2, y2 = 0, 0, 15, 15
    eps = 0.3
    x1_2, y1_2, x2_2, y2_2 = 5 - eps, 5 - eps, 5 + eps, 5 + eps

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        # empty!(ax_average)
        for k in 1:length(data_array)
            for j in 1:length(data_array[k])
                empty!(ax_master_array[k][j])
                heatmap!(ax_master_array[k][j], dim.x, dim.y, data_array[k][j][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
                mesh!(ax_master_array[k][j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(data_array[k][j][:interps]), tspan[i]))
                if signal_index != 3
                    lines!(ax_master_array[k][j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
                    lines!(ax_master_array[k][j], [x1_2, x2_2, x2_2, x1_2, x1_2], [y1_2, y1_2, y2_2, y2_2, y1_2], color=:red, linewidth=2, linestyle=:dot)
                end
            end

            idx = findfirst(tspan[i] .<= t)[1]
            empty!(ax_last_array[k])
            for j in 1:length(data_array[k])
                lines!(ax_last_array[k], t[1:idx], data_array[k][j][:signal][signal_index, 1:idx], color=colors[j])
            end

            empty!(ax_average_array[k])
            band!(ax_average_array[k], t[1:idx], lower_bound_array[k][1:idx], upper_bound_array[k][1:idx]; color = (:cyan, 0.4))
            lines!(ax_average_array[k], t[1:idx], average_array[k][1:idx], color=:blue)
            lines!(ax_average_array[k], t, gbo_ss, color=:orange)

            # band!(ax_average, t[1:idx], lower_bound_array[k][1:idx], upper_bound_array[k][1:idx]; color = (colors[k], 0.4))
            # lines!(ax_average, t[1:idx], average_array[k][1:idx], color=colors[k])
        end
    end
end

function create_average_axis(fig, fig_indices, data_array, data_description, title, signal_index, gbo_ss=nothing)
    max_bound = 0
    max_average = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        upper_bound = maximum(mean(vecs) .+ 0.5*sqrt.(var(vecs)))
        max_bound = max(max_bound, upper_bound)
        max_average = max(max_average, maximum(mean(vecs)))
    end
    
    ax_average = Axis(fig[fig_indices[1],fig_indices[2]], title = title, xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax_average, t[1], round(t[end], digits=1))
    ylims!(ax_average, 0.0, length(data_array[1]) > 1 ? max_bound * 1.20 : max_average * 1.20)
    empty!(ax_average)

    average_array = []
    std_dev_array = []
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        average = mean(vecs)
        upper_bound = average .+ 0.5*sqrt.(var(vecs))
        lower_bound = average .- 0.5*sqrt.(var(vecs))

        if length(data_array[1]) > 1
            band!(ax_average, t, lower_bound, upper_bound; color = (colors[i], 0.4))
        end
        lines!(ax_average, t, average, label="$(data_description[i])"; color = colors[i])
        
        push!(average_array, average)
        push!(std_dev_array, sqrt.(var(vecs)))
    end
    if !isnothing(gbo_ss)
        lines!(ax_average, t, gbo_ss, label="GBO"; color = colors[length(data_array)+2], linewidth = 6, linestyle = :dash)
    end
    axislegend(ax_average, position = :lt)
    return average_array, std_dev_array
end

function single_row_mpc(data_array, data_description, title, output_path, signal_index)
    fig = Figure(;size = (2000, 1600), figure_padding = (30, 50, 30, 30))
    Label(fig[1, 1:length(data_array)], title, halign = :center, fontsize = 36)

    grid_array = []
    for i in 1:length(data_array)
        push!(grid_array, Axis(fig[2, i], aspect = 1.0, title = data_description[i], xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    x1, y1, x2, y2 = 0, 0, 15, 15

    max_average = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        max_average = max(max_average, maximum(mean(vecs)))
    end
    ax_title = signal_index == 3 ? "Scattered Energy in Entire Grid" : "Scattered Energy in Top Right Quadrant"
    ax = Axis(fig[3, 1:length(data_array)], title = "", xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax, t[1], round(t[end], digits=1))
    ylims!(ax, 0.0, max_average * 1.20)
    empty!(ax)

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        empty!(ax)
        for j in 1:length(data_array)
            empty!(grid_array[j])
            heatmap!(grid_array[j], dim.x, dim.y, data_array[j][1][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(grid_array[j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(data_array[j][1][:interps]), tspan[i]))
            if signal_index != 3
                lines!(grid_array[j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
            end
            idx = findfirst(tspan[i] .<= t)[1]

            lines!(ax, t[1:idx], data_array[j][1][:signal][signal_index, 1:idx], label="$(data_description[j])"; color = colors[j])
            axislegend(ax, position = :lt)
        end
    end

end

function create_figures(data_array, data_description, title, output_path, signal_index, gbo_ss=nothing)
    fig = Figure(;size = (2000, 2000), figure_padding = (30, 50, 30, 30))
    Label(fig[1, 1:length(data_array)], title, halign = :center, fontsize = 24)
    
    max_energy = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        max_energy = max(max_energy, maximum(vcat(vecs...)))
    end

    for i in 1:length(data_array)
        ax_last = Axis(fig[2, i], title = data_description[i], xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_last, t[1], round(t[end], digits=1))
        ylims!(ax_last, 0.0, max_energy * 1.20)
        empty!(ax_last)
        for j in 1:length(data_array[i])
            lines!(ax_last, t, data_array[i][j][:signal][signal_index, :], color=colors[j])
        end
        
    end

    create_average_axis(fig, (3, 1:length(data_array)), data_array, data_description, title, signal_index, gbo_ss)
    save(output_path, fig)

    fig2 = Figure(;size = (2000, 1200), figure_padding = (30, 50, 30, 30))
    average, std_dev = create_average_axis(fig2, (1, 1), data_array, data_description, title, signal_index, gbo_ss)
    save(joinpath(dirname(output_path), "average_1-12.png"), fig2)

    return average, std_dev
end

function create_actions_plots(data_array, title_array, output_path)
    fig = Figure(;size = (2000, 1200), figure_padding = (30, 50, 30, 30))

    for i in 1:length(data_array)

        interps = data_array[i][:interps]
        actions = [interps[k].final - interps[k].initial for k in 1:env.actions]
        actions_pos = [actions[k].cylinders.pos for k in 1:env.actions]
        actions_radii = [actions[k].cylinders.r for k in 1:env.actions]
        t = [interps[k].ti for k in 1:env.actions]


        for j in 1:size(actions_pos[1])[1]
            ax = Axis(fig[i, j], title = "Time Evolution of Scatterer (#$j) Movement Actions ($(title_array[i]))", xlabel = "Time (s)", ylabel = "Position Change")# ($(mod(k, 2) == 1 ? "Δx" : "Δy"))")
            xlims!(ax, t[1], round(t[end], digits=1))
            ylims!(ax, -0.4, 0.4)
            if max(maximum([actions_pos[l][j, 1] for l in 1:env.actions]), maximum([actions_pos[l][j, 1] for l in 1:env.actions]), maximum([actions_radii[l][j, 1] for l in 1:env.actions])) >= 0.4
                ylims!(ax, -0.8, 0.8)
            end
            lines!(ax, t, [actions_pos[l][j, 1] for l in 1:env.actions], color=:red, label="Δx")
            lines!(ax, t, [actions_pos[l][j, 2] for l in 1:env.actions], color=:blue, label="Δy")
            lines!(ax, t, [actions_radii[l][j, 1] for l in 1:env.actions], color=:green, label="Δr")
            axislegend(ax, position = :lt)
        end
    end
    save(output_path, fig)
end

function create_full_single_vid(data_array, title_array, output_path, signal_index)
    fig = Figure(;size = (2400, 1600))

    vecs = [data_array[i][:signal][signal_index, :] for i in 1:length(data_array)]
    max_energy = maximum(vcat(vecs...))

    ax_energy = Axis(fig[2:5, 1], title = "Energy", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_energy, t[1], round(t[end], digits=1))
    ylims!(ax_energy, 0.0, max_energy * 1.20)

    ax_grid_arr = []
    ax_actions = []
    t_actions = []
    actions_arr = []
    for i in 1:length(data_array) 
        ax_grid = Axis(fig[(i*2-1):(i*2), 2], aspect = 1.0, title = "$(title_array[i]) ($(String(colors[i])))", xlabel = "Space (m)", ylabel = "Space (m)")
        push!(ax_grid_arr, ax_grid)

        interps = data_array[i][:interps]
        actions = [interps[k].final - interps[k].initial for k in 1:env.actions]
        actions_pos = [actions[k].cylinders.pos for k in 1:env.actions]
        t_actions = [interps[k].ti for k in 1:env.actions]
        push!(actions_arr, actions_pos)

        for j in 1:size(actions_pos[1])[1]
            for k in 1:size(actions_pos[1])[2]
                push!(ax_actions, Axis(fig[i*2-1+mod(j+1,2):i*2-1+mod(j+1,2)+1, 3 + mod(k+1, 2)], title = "Time Evolution of Scatterer (#$j) Movement Actions ($(title_array[i]))", xlabel = "Time (s)", ylabel = "Position Change ($(mod(k, 2) == 1 ? "Δx" : "Δy"))"))
                ylims!(ax_actions[end], -0.4, 0.4)
                xlims!(ax_actions[end], t_actions[1], t_actions[end])
                # lines!(ax, t, [actions_pos[l][j, k] for l in 1:env.actions])
            end
        end
    end

    # Bounding box coordinates
    x1, y1, x2, y2 = 0, 0, 15, 15

    CairoMakie.record(fig, output_path, axes(tspan[1:end-10], 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        empty!(ax_energy)
        idx = findfirst(tspan[i] .<= t)[1]
        idx2 = findfirst(tspan[i] .<= t_actions)[1]
        for k in 1:length(data_array)

            empty!(ax_grid_arr[k])
            heatmap!(ax_grid_arr[k], dim.x, dim.y, data_array[k][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax_grid_arr[k], Waves.multi_design_interpolation(Vector{DesignInterpolator}(data_array[k][:interps]), tspan[i]))
            if signal_index != 3
                lines!(ax_grid_arr[k], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
            end

            lines!(ax_energy, t[1:idx], data_array[k][:signal][signal_index, 1:idx], color=colors[k])
            
            for j in 1:length(actions_arr)
                empty!(ax_actions[j*2-1])
                lines!(ax_actions[j*2-1], t_actions[1:idx2], [actions_arr[j][i][1] for i in 1:env.actions][1:idx2], color=:blue)
                empty!(ax_actions[j*2])
                lines!(ax_actions[j*2], t_actions[1:idx2], [actions_arr[j][i][2] for i in 1:env.actions][1:idx2], color=:blue)
            end
        end

    end

end

# dataset_name = "pos_adjustment_masked_signals_M=1"
dataset_name = "full_adjustment_masked_signals_M=2"
DATA_PATH = "scratch/$dataset_name"
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
dim = env.dim

env.actions = 200
t = build_tspan(0.0f0, env.dt, env.actions * env.integration_steps)
seconds = 40.0
frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
tspan = collect(range(t[1], t[end], frames))

folders = [ "M=1_pos_focusing_shots=512_noOverlap_2.mpc", 
            "M=1_pos_suppression_shots=512.mpc",
            "M=2_pos_focusing_shots=512_noOverlap.mpc",
            "M=2_pos_suppression_shots=512_noOverlap.mpc",
            "M=4_pos_focusing_shots=1600_noOverlap.mpc",
            "M=4_pos_suppression_shots=512_noOverlap.mpc",
            "M=2_fullyAdj_focusing_shots=1024_noOverlap.mpc",
            "M=2_fullyAdj_suppression_shots=1024_noOverlap.mpc" 
            ]



my_theme = Theme(fontsize = 26)
set_theme!(my_theme)

function log_message(logpath, message)
    mkpath(logpath)
    logfile = open(joinpath(logpath, "log2.txt"), "a")
    println(logfile, "$message")
    close(logfile)
end

# for folder in folders
#     log_message("initial_conditions", folder)
#     runs = BSON.load(joinpath(folder, "positions.bson"))[:pos]
#     for (i, run) in enumerate(runs)
#         M = length(run.cylinders.r)
#         log_message("initial_conditions", "Initial Conditions for run #$i:")
#         for j in 1:M
#             log_message("initial_conditions", "Cylinder $j Position: ($(run.cylinders.pos[j, 1]), $(run.cylinders.pos[j, 2]))")
#             log_message("initial_conditions", "Cylinder $j Radius: $(run.cylinders.r[j])")
#         end
#         log_message("initial_conditions", "")
#     end
#     log_message("initial_conditions", "")
# end


results = []
fig2 = Figure(;size = (1400, 1400), figure_padding = (30, 50, 30, 30))
for (k, output_folder) in []#enumerate(folders[[7 8]])
    # for (k, output_folder) in enumerate(folders)
    # output_folder = folders[7]

    focusing = contains(output_folder, "focus")

    prefix = ""
    title = focusing ? "Focusing" : "Suppression"
    signal_index = focusing ? 13 : 3
    j=1
    step = 12
    # data_description = ["AEM", "GBO (Set 1)", "GBO (Set 2)"]#, "GBO (9, -9)"]
    # data_description = ["AEM", "NODE", "Random", "GBO"]
    data_description = ["AEM", "NODE", "Random"]
    index_array = vcat(j:(j+step-1))
    @time mpc_data = [BSON.load(joinpath(output_folder, "mpc_$i.bson")) for i in index_array]
    @time node_data = [BSON.load(joinpath(output_folder, "node_$i.bson")) for i in index_array]
    @time random_data = [BSON.load(joinpath(output_folder, "random_$i.bson")) for i in index_array]
    data_array = [mpc_data, node_data, random_data]
    
    gbo_files = [x for x in readdir(output_folder) if contains(x, "gbo_set")]
    gbo = [BSON.load(joinpath(output_folder, gbo_files[i])) for i in 1:length(gbo_files)]

    # @time gbo_2 = [BSON.load(joinpath(output_folder, "gbo_set2_$i.bson")) for i in 1:11]
    # @time gbo_1 = [BSON.load(joinpath(output_folder, "gbo_set1_$i.bson")) for i in 1:11]
    # gbo_ss = get_best_gbo([gbo_1, gbo_2], signal_index)
    gbo_ss = get_best_gbo([gbo], signal_index)
# data_array = [mpc_data, node_data, random_data, gbo_1[index_array]]

# @time average_array, std_dev_array = create_figures(data_array, data_description, title, joinpath(output_folder, "$(prefix)$j-$(j+step-1)_GBO.png"), focusing ? 13 : 3, gbo_ss)

# data_array = [mpc_data, node_data, gbo_1[index_array]]
# data_description = ["AEM", "NODE", "GBO"]
# @time multiple_mpc_rendering(data_array, data_description, title, joinpath(output_folder, "$(prefix)$j-$(j+step-1).mp4"), focusing ? 13 : 3, gbo_ss)
# single_row_mpc(data_array, data_description, "Fully Adjustable Scatterers (Position and Radii, 2 Scatterers) ", joinpath(output_folder, "single_row3.mp4"), signal_index)

# @time average_array, std_dev_array = create_figures(data_array, data_description, title, joinpath(output_folder, "$(prefix)$j-$(j+step-1)_GBO.png"), focusing ? 13 : 3, gbo_ss)
    
# step = 6
# for j in [1 7]
#     @time multiple_mpc_rendering([data_array[i][j:j+step-1] for i in 1:length(data_array)], data_description, title, joinpath(output_folder, "$(prefix)$j-$(j+step-1)_GBO.mp4"), focusing ? 13 : 3, gbo_ss)
# end
    # push!(results, output_folder, average_array, std_dev_array)

    if focusing
        create_average_axis(fig2, (k, 1), data_array, data_description, title, focusing ? 13 : 3)
    else
        create_average_axis(fig2, (k, 1), data_array, data_description, title, focusing ? 13 : 3, gbo_ss)
    end
    
end

# save("1314.png", fig2)


# contents = ["actions", "1-6.mp4", "1-12.png", "7-12.mp4", "average_1-12.png"]
# out = mkpath("average_plots")
# for folder in folders
#     for c in contents[[3, 5]]
#         pos = first(findfirst("_shot", folder))
#         if contains(c, ".")
#             cp(joinpath(folder, c), joinpath(out, "$(folder[1:pos-1])_$c"); force=true)
#         else
#             mkpath(joinpath(out, "actions_$(folder[1:pos-1])"))
#             files = readdir(joinpath(folder, "actions"))
#             for file in files
#                 cp(joinpath(folder, c, file), joinpath(out, "actions_$(folder[1:pos-1])", file); force=true)
#             end
#         end
#     end
# end



# output_folder = "M=2_fullyAdj_focusing_shots=1024_noOverlap.mpc"
# # output_folder = "M=4_pos_focusing_shots=1600_noOverlap.mpc"
# focusing = contains(output_folder, "focus")
# title = "$(focusing ? "Focused Energy in Top Right Quadrant" : "Suppressed Energy in Entire Grid")"

# j=1
# step = 12
# data_description = ["AEM", "NODE", "Random"]
# index_array = vcat(j:(j+step-1))
# @time mpc_data = [BSON.load(joinpath(output_folder, "mpc_$i.bson")) for i in index_array]
# @time node_data = [BSON.load(joinpath(output_folder, "node_$i.bson")) for i in index_array]
# @time random_data = [BSON.load(joinpath(output_folder, "random_$i.bson")) for i in index_array]
# data_array = [mpc_data, node_data, random_data]

# j = 1
# step = 12

# fig = Figure(;size = (1000, 600), figure_padding = (30, 50, 30, 30))
# fig_indices = (1, 1)
# signal_index = focusing ? 13 : 3

# max_bound = 0
# max_average = 0
# for i in 1:length(data_array)
#     vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
#     upper_bound = maximum(mean(vecs) .+ 0.5*sqrt.(var(vecs)))
#     max_bound = max(max_bound, upper_bound)
#     max_average = max(max_average, maximum(mean(vecs)))
# end

# ax_average = Axis(fig[fig_indices[1],fig_indices[2]], title = "Mean of $(signal_index == 3 ? "Suppressed Energy in Entire Grid" : "Focused Energy in Top Right Quadrant")", xlabel = "Time (s)", ylabel = "Energy")
# xlims!(ax_average, t[1], round(t[end], digits=1))
# ylims!(ax_average, 0.0, max_bound * 1.20)
# empty!(ax_average)

# average_array = []
# std_dev_array = []
# for i in 1:length(data_array)
#     vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
#     average = mean(vecs)
#     upper_bound = average .+ 0.5*sqrt.(var(vecs))
#     lower_bound = average .- 0.5*sqrt.(var(vecs))

#     band!(ax_average, t, lower_bound, upper_bound; color = (colors[i], 0.4))
#     lines!(ax_average, t, average, label="$(data_description[i])"; color = colors[i])
    
#     push!(average_array, average)
#     push!(std_dev_array, sqrt.(var(vecs)))
# end
# axislegend(ax_average, position = :lt)

# save("11.png", fig)

# for run in 1:12
# # run = 5
#     fig = Figure(;size = (1800, 600))
#     color_max = 0
#     x1, y1, x2, y2 = 0, 0, 15, 15
#     data_array = [mpc_data[run:run], node_data[run:run], random_data[run:run]]
#     for i in 1:length(data_array)
#         mid = ceil(Int, length(data_array[i][1][:x]) / 2)
#         ax = Axis(fig[1, i], aspect = 1.0, title="$(focusing ? "Focused" : "Suppressed") Steady State Scattered Energy \n$(data_description[i])")#, xlabel = "Space (m)", ylabel = "Space (m)")
#         # mean_matrix = mean(mean([data_array[i][j][:x][mid:end] for j in length(data_array[i])]))
#         mean_matrix = mean(mean([data_array[i][j][:x][930:930] for j in length(data_array[i])]))
#         hm = heatmap!(ax, dim.x, dim.y, mean_matrix, colormap=:viridis)
#         if focusing
#             lines!(ax, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
#         end
#         color_max = max(color_max, maximum(mean_matrix))

#         interps = data_array[i][1][:interps]
#         arrows = [(interps[k].initial.cylinders.pos, interps[k].final.cylinders.pos)  for k in 1:env.actions]
#         for scatterer_index in 1:size(arrows[1][1], 1)
#             x, y, u, v = [], [], [], []
#             for arrow in arrows[end:end]
#                 push!(x, arrow[1][scatterer_index, 1])
#                 push!(y, arrow[1][scatterer_index, 2])
#                 push!(u, arrow[2][scatterer_index, 1] - arrow[1][scatterer_index, 1])
#                 push!(v, arrow[2][scatterer_index, 2] - arrow[1][scatterer_index, 2])
#             end
#             strength = vec(sqrt.(u .^ 2 .+ v .^ 2))
#             # Makie.arrows!(x, y, u, v, arrowcolor = strength, linecolor = strength, arrowsize = 15)
#         end
#     end


#     Colorbar(fig[1, length(data_array)+1], label = "Scattered Energy", colormap=:viridis, colorrange=(0, color_max))
#     save("$run.png", fig)
# end


# interps = data_array[1][1][:interps]
# arrows = [(interps[k].initial.cylinders.pos, interps[k].final.cylinders.pos)  for k in 1:env.actions]

# x = []
# y = []
# u = []
# v = []
# for arrow in arrows[101:200]
#     push!(x, arrow[1][1, 1])
#     push!(y, arrow[1][1, 2])
#     push!(u, arrow[2][1, 1] - arrow[1][1, 1])
#     push!(v, arrow[2][1, 2] - arrow[1][1, 2])
# end
# strength = vec(sqrt.(u .^ 2 .+ v .^ 2))
# ax = fig.content[3]

# Makie.arrows!(x, y, u, v, arrowcolor = strength, linecolor = strength)



# fig = Figure(;size = (2000, 1200), figure_padding = (30, 50, 30, 30))

# for i in 1:length(data_array)

#     interps = data_array[i][1][:interps]
#     actions = [interps[k].final - interps[k].initial for k in 1:env.actions]
#     actions_pos = [actions[k].cylinders.pos for k in 1:env.actions]
#     actions_radii = [actions[k].cylinders.r for k in 1:env.actions]
#     t = [interps[k].ti for k in 1:env.actions]
#     arrows = [(interps[k].initial.cylinders.pos, interps[k].final.cylinders.pos) for k in 1:env.actions]


#     for j in 1:size(actions_pos[1])[1]
#         ax = Axis(fig[i, j], title = "Time Evolution of Scatterer (#$j) Movement Actions ($(title_array[i]))", xlabel = "Time (s)", ylabel = "Position Change")# ($(mod(k, 2) == 1 ? "Δx" : "Δy"))")
#         xlims!(ax, t[1], round(t[end], digits=1))
#         ylims!(ax, -0.4, 0.4)
#         if max(maximum([actions_pos[l][j, 1] for l in 1:env.actions]), maximum([actions_pos[l][j, 1] for l in 1:env.actions]), maximum([actions_radii[l][j, 1] for l in 1:env.actions])) >= 0.4
#             ylims!(ax, -0.8, 0.8)
#         end
#         lines!(ax, t, [actions_pos[l][j, 1] for l in 1:env.actions], color=:red, label="Δx")
#         lines!(ax, t, [actions_pos[l][j, 2] for l in 1:env.actions], color=:blue, label="Δy")
#         lines!(ax, t, [actions_radii[l][j, 1] for l in 1:env.actions], color=:green, label="Δr")
#         axislegend(ax, position = :lt)
#     end
# end
# save(output_path, fig)



# for f in readdir()
#     if startswith(f, "actions")
#         for ff in readdir(f)
#             new_name = "$(f)_$(ff[end-5:end])"
#             cp(joinpath(f, ff), joinpath(mkpath("actions"), new_name))
#     end
# end

# for i in 1:3:length(results)
#     println(results[i])
#     middle_index = ceil(Int, length(results[i+1][1]) / 2)
#     aem_average = mean(results[i+1][1][middle_index:end])
#     node_average = mean(results[i+1][2][middle_index:end])
#     random_average = mean(results[i+1][3][middle_index:end])
#     aem_std_dev = mean(results[i+2][1][middle_index:end])
#     node_std_dev = mean(results[i+2][2][middle_index:end])
#     random_std_dev = mean(results[i+2][3][middle_index:end])

#     println("AEM: $(round(aem_average, digits=2)) ± $(round(0.5*aem_std_dev, digits=2))")
#     println("NODE: $(round(node_average, digits=2)) ± $(round(0.5*node_std_dev, digits=2))")
#     println("Random: $(round(random_average, digits=2)) ± $(round(0.5*random_std_dev, digits=2))")
#     println()
# end

# folder = "M=2_fullyAdj_focusing_shots=1024_noOverlap.mpc"
# for f in readdir(folder)
#     if startswith(f, "gbo")
#         pos = first(findfirst("_", f)) + 1
#         new_name = "gbo_set3_$(f[pos:end])"
#         mv(joinpath(folder, f), joinpath(folder, new_name))
#     end
# end

function focus_vs_suppress(focus, suppress, data_description, title, output_path, gbo_ss=nothing)
    fig = Figure(;size = (1100, 1400), figure_padding = (30, 50, 30, 30))

    max_energy_focus = maximum(vcat([focus[i][:signal][13, :] for i in 1:length(focus)]...))
    max_energy_suppress = maximum(vcat([suppress[i][:signal][3, :] for i in 1:length(focus)]...))

    ax_grid_focus = []
    for i in 1:length(focus) 
        push!(ax_grid_focus, Axis(fig[1, i], aspect = 1.0, title = data_description[i], xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    ax_focus = Axis(fig[2, 1:length(focus)], title = "Focusing", xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax_focus, t[1], round(t[end], digits=1))
    ylims!(ax_focus, 0.0, max_energy_focus * 1.20)

    ax_grid_suppress = []
    for i in 1:length(suppress) 
        push!(ax_grid_suppress, Axis(fig[3, i], aspect = 1.0, title = data_description[i], xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    ax_suppress = Axis(fig[4, 1:length(suppress)], title = "Suppression", xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax_suppress, t[1], round(t[end], digits=1))
    ylims!(ax_suppress, 0.0, max_energy_suppress * 1.20)

    x1, y1, x2, y2 = 0, 0, 15, 15

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        empty!(ax_focus)
        empty!(ax_suppress)
        idx = findfirst(tspan[i] .<= t)[1]
        for k in 1:3
            empty!(ax_grid_focus[k])
            heatmap!(ax_grid_focus[k], dim.x, dim.y, focus[k][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax_grid_focus[k], Waves.multi_design_interpolation(Vector{DesignInterpolator}(focus[k][:interps]), tspan[i]))
            lines!(ax_grid_focus[k], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)

            empty!(ax_grid_suppress[k])
            heatmap!(ax_grid_suppress[k], dim.x, dim.y, suppress[k][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax_grid_suppress[k], Waves.multi_design_interpolation(Vector{DesignInterpolator}(suppress[k][:interps]), tspan[i]))
            
            lines!(ax_focus, t[1:idx], focus[k][:signal][13, 1:idx], label=data_description[k], color=colors[k])
            lines!(ax_suppress, t[1:idx], suppress[k][:signal][3, 1:idx], label=data_description[k],  color=colors[k])
        end
        lines!(ax_suppress, t, gbo_ss, label="GBO"; color = colors[5], linewidth = 4, linestyle = :dash)
        axislegend(ax_focus, position = :lt)
        axislegend(ax_suppress, position = :lt)
    end
end


folder_f = folders[7]
folder_s = folders[8]
prefixes = ["mpc", "node", "random"]
data_description = ["AEM", "NODE", "Random"]

# @time focus = [BSON.load(joinpath(folder_f, "$(prefixes[i])_1.bson")) for i in 1:length(prefixes)]
# @time suppress = [BSON.load(joinpath(folder_s, "$(prefixes[i])_1.bson")) for i in 1:length(prefixes)]
# gbo_files = [x for x in readdir(folder_s) if contains(x, "gbo_set")]
# gbo = [BSON.load(joinpath(folder_s, gbo_files[i])) for i in 1:length(gbo_files)]
# gbo_ss = get_best_gbo([gbo], 3)

output_path = "focus_vs_suppress.mp4"
title = ""


my_theme = Theme(fontsize = 22)
set_theme!(my_theme)


focus_vs_suppress(focus, suppress, data_description, title, output_path, gbo_ss)

