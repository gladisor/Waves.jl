using Waves, CairoMakie, Flux, BSON
using Optimisers
using Images: imresize
println("Loaded Packages")

my_theme = Theme(fontsize = 34)
set_theme!(my_theme)

datasets = ["pos_adjustment_masked_signals_M=1", "pos_adjustment_masked_signals_M=2", "pos_adjustment_masked_signals_M=4", "full_adjustment_masked_signals_M=2"]
run_tuples = [  (42777, 42890, true, datasets[1]), (42722, 42889, false, datasets[1]), (42694, 42884, true, datasets[2]), (42774, 42891, false, datasets[2]), # M=1, M=2
                (42880, 42883, true, datasets[3]), (42881, 42887, false, datasets[3]), (42873, 42886, true, datasets[4]), (42878, 42887, false, datasets[4])] # M=4, M=2(fullyAdj)
                # AEM_jobid, NODE_jobid, focusing(true/false), dataset_name

for (jobid, node_jobid, focusing, dataset_name) in run_tuples[3:3]
    M = dataset_name[end]
    DATA_PATH = "scratch/$dataset_name"
    for checkpoint in [8000]
        model_name = "AEM_$(focusing ? "focusing" : "suppression")_jobID=$jobid"
        node_model_name = "NODE_$(focusing ? "focusing" : "suppression")_jobID=$node_jobid"

        ## generating paths
        MODEL_PATH = joinpath(DATA_PATH, "models/$model_name/checkpoint_step=$checkpoint/checkpoint.bson")
        NODE_MODEL_PATH = joinpath(DATA_PATH, "models/$node_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
        ## loading from storage
        model = BSON.load(MODEL_PATH)[:model]
        node_model = BSON.load(NODE_MODEL_PATH)[:model]
        for s_i in [1]
            for i in 477 #475:480 #[498 500]
                ## loading data
                episode_number = i #497
                ep = Episode(path = joinpath(DATA_PATH, "episodes/episode$episode_number.bson"))
                horizon = 100
                s, a, t, y = Flux.batch.(prepare_data(ep, horizon))
                start_index = s_i
                s = s[start_index:end]
                a = a[:, start_index:end]
                t = t[:, start_index:end]
                y = y[:, :, start_index:end]
                ## inferrence
                @time y_hat = model(s[1, :], a[:, [1]], t[:, [1]])
                @time y_hat_node = node_model(s[1, :], a[:, [1]], t[:, [1]])
                ## plotting comparison
                fig = Figure(;size = (1600, 1200), figure_padding = (10, 30, 10, 10))
                ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "$(focusing ? "Focused" : "Suppressed") Energy", title = "$(focusing ? "Focused" : "Suppressed") Energy Prediction With Random Control")
                # ylims!(ax, -0.4, 0.4)
                xlims!(ax, 0, t[end, 1])

                lines!(ax, t[:, 1], y[:, focusing ? 13 : 3, 1], label = "Ground Truth")
                lines!(ax, t[:, 1], y_hat[:, 3, 1], color = (:green, 0.6), label = "AEM")
                lines!(ax, t[:, 1], y_hat_node[:, 3, 1], color = (:red, 0.6), label = "NODE")
                axislegend(ax, position = :lt)
                output_folder = mkpath(joinpath("predictions2", "M=$(dataset_name[end])$(contains(dataset_name, "full") ? "fullyAdj" : "")_$(focusing ? "focus" : "suppress")"))
                save(joinpath(output_folder, "$(checkpoint)_$(episode_number)_$(start_index).png"), fig)
            end
        end
    end
end
