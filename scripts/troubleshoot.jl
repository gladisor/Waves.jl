using Waves, Flux, CairoMakie, BSON, Statistics

function multiple_mpc_rendering(data_array, data_description, title, output_path)

    colors = [  :red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown, :pink,
                :lime, :teal, :violet, :gold, :indigo, :olive, :navy, :coral, :turquoise, :salmon ]
    fig = Figure(;size = (2400, 1600))

    max_bound = 0
    max_energy = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][3, :] for j in 1:length(data_array[i])]
        max_energy = max(max_energy, maximum(vcat(vecs...)))
        upper_bound = maximum(mean(vecs) .+ sqrt.(var(vecs)))
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
        xlims!(ax_last, t[1], t[end])
        ylims!(ax_last, 0.0, max_energy * 1.20)
        
        push!(ax_master_array, ax_arr)
        push!(ax_last_array, ax_last)

        vecs = [data_array[i][j][:signal][3, :] for j in 1:length(data_array[i])]
        push!(average_array, mean(vecs))
        push!(upper_bound_array, average_array[end] .+ sqrt.(var(vecs)))
        push!(lower_bound_array, average_array[end] .- sqrt.(var(vecs)))

        ax_average = Axis(fig[(div(length(data_array[i]), 2) + 1):length(data_array[i]), 2*i], title = "Mean and Standard Deviation ($(data_description[i]))", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_average, t[1], t[end])
        ylims!(ax_average, 0.0, max_bound * 1.20)
        push!(ax_average_array, ax_average)
    end

    # Bounding box coordinates
    x1, y1 = 0, 0
    x2, y2 = 15, 15

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        for k in 1:length(data_array)
            for j in 1:length(data_array[k])
                empty!(ax_master_array[k][j])
                heatmap!(ax_master_array[k][j], dim.x, dim.y, data_array[k][j][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
                mesh!(ax_master_array[k][j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(data_array[k][j][:interps]), tspan[i]))
                lines!(ax_master_array[k][j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
            end

            idx = findfirst(tspan[i] .<= t)[1]
            empty!(ax_last_array[k])
            for j in 1:length(data_array[k])
                lines!(ax_last_array[k], t[1:idx], data_array[k][j][:signal][3, 1:idx], color=colors[j])
            end

            empty!(ax_average_array[k])
            band!(ax_average_array[k], t[1:idx], lower_bound_array[k][1:idx], upper_bound_array[k][1:idx], color=:cyan, transparency=0.5)
            lines!(ax_average_array[k], t[1:idx], average_array[k][1:idx], color=:blue)
        end
    end
end

function moving_average(signal, window_size)
    return [mean(signal[max(1, i-div(window_size, 2)+1):min(i+div(window_size, 2), length(signal))]) for i in 1:length(signal)]
end

function moving_std(signal, window_size)
    return [std(signal[max(1, i-window_size+1):i]) for i in 1:length(signal)]
end

function create_figures(data_array, data_description, title, output_path)
    colors = [  :red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown, :pink,
                :lime, :teal, :violet, :gold, :indigo, :olive, :navy, :coral, :turquoise, :salmon ]
    fig = Figure(;size = (1600, 2400))

    max_bound = 0
    max_energy = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][3, :] for j in 1:length(data_array[i])]
        max_energy = max(max_energy, maximum(vcat(vecs...)))
        upper_bound = maximum(mean(vecs) .+ sqrt.(var(vecs)))
        max_bound = max(max_bound, upper_bound)
    end

    for i in 1:length(data_array)
        ax_last = Axis(fig[1:div(size(data_array[1], 1), 2)  , i], title = "$title ($(data_description[i]))", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_last, t[1], t[end])
        ylims!(ax_last, 0.0, max_energy * 1.20)

        empty!(ax_last)
        for j in 1:length(data_array[i])
            window_size = 4000
            std_deviation = moving_std(data_array[i][j][:signal][3, :], window_size)
            upper_signal = moving_average(data_array[i][j][:signal][3, :], window_size) .+ std_deviation
            lower_signal = moving_average(data_array[i][j][:signal][3, :], window_size) .- std_deviation

            # band!(ax_last, t, lower_signal, upper_signal, color=colors[end-j])
            lines!(ax_last, t, data_array[i][j][:signal][3, :], color=colors[j])
        end
        
        vecs = [data_array[i][j][:signal][3, :] for j in 1:length(data_array[i])]
        average = mean(vecs)
        upper_bound = average .+ sqrt.(var(vecs))
        lower_bound = average .- sqrt.(var(vecs))

        ax_average = Axis(fig[(div(size(data_array[1], 1), 2)+1):size(data_array[1], 1), i], title = "Mean and Standard Deviation ($(data_description[i]))", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_average, t[1], t[end])
        ylims!(ax_average, 0.0, max_bound * 1.20)
        empty!(ax_average)
        band!(ax_average, t, lower_bound, upper_bound, color=:cyan, transparency=0.5)
        lines!(ax_average, t, average, color=:blue)

    end

    save(output_path, fig)
end


dataset_name = "pos_adjustment_masked_M=2"
# dataset_name = "dataset_pos_adjustment_masked"
DATA_PATH = "scratch/$dataset_name"
@time env = BSON.load(joinpath(DATA_PATH, "env_4.bson"))[:env]
dim = env.dim

env.actions = 200
t = build_tspan(0.0f0, env.dt, env.actions * env.integration_steps)
seconds = 40.0
frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
tspan = collect(range(t[1], t[end], frames))



step = 3
# output_folder = "42526_AEM_42525_NODE_10000_focus.mpc"
# output_folder = "42527_AEM_42606_NODE_6000_minScattering.mpc"
# output_folder = "42616_AEM_42617_NODE_6000_focus.mpc"
# output_folder = "42616_AEM_42617_NODE_8700_focus_opposite_quadrant_initialization.mpc"
# output_folder = "42616_AEM_42617_NODE_9000_focus_opposite_quadrant_initialization.mpc"
output_folder = "42526_AEM_10000_focus_opposite_quadrant_initialization.mpc"
# output_folder = "42616_AEM_11000_focus_opposite_quadrant_initialization.mpc"

prefix = "comparison"
# title = "Total Scattered Energy"
title = "Focused Energy"# in Upper Right Quadrant"


j=1
step = 8
mpc_data_1_1024 = [BSON.load(joinpath(output_folder, "mpc_horizon=1_shots=1024_$i.bson")) for i in j:(j+step-1)]
mpc_data_1_512 = [BSON.load(joinpath(output_folder, "mpc_horizon=1_shots=512_$i.bson")) for i in j:(j+step-1)]
mpc_data_1 = [BSON.load(joinpath(output_folder, "mpc_horizon=1_$i.bson")) for i in j:(j+step-1)]
# mpc_data_2 = [BSON.load(joinpath(output_folder, "mpc_horizon=2_$i.bson")) for i in j:(j+step-1)]
# mpc_data_5 = [BSON.load(joinpath(output_folder, "mpc_horizon=5_$i.bson")) for i in j:(j+step-1)]
# node_data = [BSON.load(joinpath(output_folder, "node_$i.bson")) for i in j:(j+step-1)]
random_data = [BSON.load(joinpath(output_folder, "random_horizon=1_$i.bson")) for i in j:(j+step-1)]
# data_array = [mpc_data, node_data, random_data]
# data_array = [mpc_data_1_1024, mpc_data_1_512, mpc_data_1, mpc_data_2, mpc_data_5, random_data]
data_array = [mpc_data_1_1024, mpc_data_1_512, mpc_data_1, random_data]
multiple_mpc_rendering(data_array, ["MPC Shots=1024", "MPC Shots=512", "MPC Shots=256", "Random"], title, joinpath(output_folder, "$(prefix)_$j-$(j+step-1).mp4"))
# create_figures([mpc_data, node_data, random_data], ["MPC", "NODE", "Random"], title, joinpath(output_folder, "$(prefix)_$j-$(j+step-1).png"))
# create_figures(data_array, ["MPC Horizon=1 ; Shots=1024", "Random"], title, joinpath(output_folder, "$(prefix)_$j-$(j+step-1).png"))
# create_figures(data_array, ["MPC Horizon=1 ; Shots=1024", "MPC Horizon=1 ; Shots=512", "MPC Horizon=1 ; Shots=256", "MPC Horizon=2 ; Shots=256", "MPC Horizon=5 ; Shots=256", "Random"], title, joinpath(output_folder, "$(prefix)_$j-$(j+step-1).png"))

