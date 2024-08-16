# using Waves
# using CairoMakie

# # wave = ep.s[end].wave
# # dim = ep.s[end].dim
# # fig = Figure()
# # ax = Axis(fig[1, 1])
# # heatmap!(ax, dim.x, dim.y, wave[:, :, 1])
# # # mesh!(ax, ep.s[end].design)

# # # save("mesh.png", fig)

# dataset = "scratch/dataset_pos_adjustment_masked"
# model_path = "AEM_horizon=1_batchsize=256_nfreq=500_jobID=41953"
# checkpoint = 20

# ep = Episode(path = "$dataset/episodes/episode1.bson")
# model = BSON.load("$dataset/models/$model_path/checkpoint_step=$checkpoint/checkpoint.bson")[:model]

# s, a, t, y, s_ = prepare_data([ep], 1)

# matrix_array = []

# for checkpoint in 6000:20:6960
#     model = BSON.load("$dataset/models/$model_path/checkpoint_step=$checkpoint/checkpoint.bson")[:model]

#     coefficient_matrix = s |> model.wave_encoder.base |> model.wave_encoder.head[1:2]
#     coefficient_matrix = coefficient_matrix[:, 1:4, 180]

#     push!(matrix_array, coefficient_matrix)
# end


# num_matrices = length(matrix_array)
# matrix_size = size(matrix_array[1])  # Assuming all matrices have the same size
# variance_map = zeros(matrix_size)
# cv_map = zeros(matrix_size)

# for i in 1:matrix_size[1]
#     for j in 1:matrix_size[2]
#         element_values = [matrix_array[k][i, j] for k in 1:num_matrices]
#         variance_map[i, j] = var(element_values)
#         element_mean = mean(element_values)

#         if element_mean ≈ 0
#             cv_map[i, j] = NaN  # Or any other appropriate handling 
#         else
#             cv_map[i, j] = std(element_values) / element_mean
#         end
#     end
# end


# threshold = 0.1
# high_cv_indices = findall(x -> x > threshold, cv_map)
# println("Indices with variance above threshold:")
# display(high_variance_indices)








# using Waves, CairoMakie, Flux, BSON, CUDA
# using Optimisers
# using Images: imresize
# using Statistics, LinearAlgebra
# Flux.CUDA.allowscalar(false)
# println("Loaded Packages")
# Flux.device!(0)
# display(Flux.device())

# dataset_name = "dataset_pos_adjustment_masked"
# DATA_PATH = "scratch/$dataset_name"
# @time data = [Episode(path = joinpath(DATA_PATH, "episodes/episode$i.bson")) for i in 451:500]

# function create_new_model()
#     activation = leakyrelu
#     h_size = 256
#     in_channels = 4
#     nfreq = 500
#     elements = 1024 
#     latent_gs = 100.0f0
#     pml_width = 10.0f0
#     pml_scale = 10000.0f0
#     latent_dim = OneDim(latent_gs, elements)
#     @time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
#     @time model = gpu(AcousticEnergyModel(;env, h_size, in_channels, nfreq, pml_width, pml_scale, latent_dim, base_function=build_cnn_base))
# end

# function create_eigenvalues_plot(model, data; runs::Int = 20, max_horizon::Int = 20, batchsize::Int = 32, path::String = "eig_val_x.png")
#     data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

#     sum_array = zeros(Float32, max_horizon)
#     @time for run in 1:runs
#         values = Vector{Float32}()
#         @time for h in 1:max_horizon
#             loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
#             s, a, t, y = gpu(Flux.batch.(first(loader)))
            
#             z0, θ = get_parameters_and_initial_condition(model, s, a, t)
#             loss, back = Flux.pullback(z0) do _z0
#                 z = model.iter(_z0, t, θ)
#                 y_hat = compute_latent_energy(z, model.dx)
#                 return Flux.mse(y_hat, y[:,1:size(y_hat, 2),:])
#             end
            
#             gs = back(one(loss))[1]
            
#             gs_3 = gs[:,3,:]
#             # det(cov(cpu(gs_3)'))
#             push!(values, eigvals(cov(cpu(gs_3)'))[end])
#         end
#         println("Run #$run done.")
#         sum_array = sum_array .+ values
#     end
#     averaged_values = sum_array / runs

#     fig = Figure()
#     ax = Axis(fig[1, 1], xlabel = "Horizon (#actions)", ylabel = "max(Eigenvalues)")
#     lines!(ax, collect(1:max_horizon), averaged_values)
#     save(path, fig)
# end

# function create_cov_heatmap(model, data; runs::Int = 20, max_horizon::Int = 20, batchsize::Int = 32, path::String = "")
#     data_loader_kwargs = Dict(:batchsize => batchsize, :shuffle => true, :partial => false)

#     cov_matrices = []
#     @time for h in 1:max_horizon
#         loader = Flux.DataLoader(prepare_data(data, h); data_loader_kwargs...)
#         s, a, t, y = gpu(Flux.batch.(first(loader)))
        
#         z0, θ = get_parameters_and_initial_condition(model, s, a, t)
#         loss, back = Flux.pullback(θ) do _θ
#             z = model.iter(z0, t, _θ)
#             return z[:,:,:,end]
#         end
        
#         one_loss = CUDA.fill(1f0, size(loss))
#         gs = back(one_loss)[1]
#         # gs_3 = gs[:,3,:]
#         gs_3 = gs[3]
#         push!(cov_matrices, cov(cpu(gs_3)'))

#     end

#     for i in 1:size(cov_matrices, 1)
#         fig = Figure()
#         ax = Axis(fig[1, 1])
#         hm = heatmap!(ax, cov_matrices[i])
#         Colorbar(fig[1, 2], hm)
#         save(joinpath(path, "$i.png"), fig)
#     end
# end


# jobid = 42204
# checkpoints = [7600]
# model_path = "AEM_batchsize=32_jobID=$jobid"

# output_path = mkpath("$jobid.cov")
# for checkpoint in checkpoints
#     local model = gpu(BSON.load("$DATA_PATH/models/$model_path/checkpoint_step=$checkpoint/checkpoint.bson")[:model])
#     create_cov_heatmap(model, data; runs = 1, max_horizon = 20, path=mkpath("$(output_path)/$(checkpoint)"))
# end

# # using Waves
# # using CairoMakie
# # dataset = "scratch/dataset_pos_adjustment_masked"
# # ep = Episode(path = "$dataset/episodes/episode1.bson")

# # wave = ep.s[end].wave
# # dim = ep.s[end].dim
# # fig = Figure()
# # ax = Axis(fig[1, 1])
# # # heatmap!(ax, dim.x, dim.y, wave[:, :, 1])
# # heatmap!(ax, dim.x, dim.y, patches[7] .* wave[:,:,1])

# # mesh!(ax, ep.s[end].design)

# # save("mesh.png", fig)

# dataset = "scratch/dataset_pos_adjustment_masked"
# ep = Episode(path = "$dataset/episodes/episode1.bson")

using Waves, Flux, CairoMakie, BSON, Statistics

dataset_name = "pos_adjustment_masked_M=2"
DATA_PATH = "scratch/$dataset_name"
@time env = BSON.load(joinpath(DATA_PATH, "env_4.bson"))[:env]
dim = env.dim

jobid = 42474
model_name = "AEM_batchsize=64_jobID=$jobid"
checkpoint_step = 6000

MODEL_PATH = "scratch/$dataset_name/models/$model_name/checkpoint_step=$checkpoint_step/checkpoint.bson"
env.actions = 200
t = build_tspan(0.0f0, env.dt, env.actions * env.integration_steps)
seconds = 40.0
frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
tspan = collect(range(t[1], t[end], frames))

function multiple_mpc_rendering(mpc_data, random_data, output_path)
    colors = [  :red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown, :pink,
                :lime, :teal, :violet, :gold, :indigo, :olive, :navy, :coral, :turquoise, :salmon ]
    fig = Figure(;size = (1600, 1600))
    ax_arr = []
    for i in 1:size(mpc_data, 1)
        push!(ax_arr, Axis(fig[i, 1], aspect = 1.0, title = "$(String(colors[i]))", xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    ax_last = Axis(fig[1:div(size(mpc_data, 1), 2)  , 2], title = "Minimized Energy in Upper Right Quadrant (MPC)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_last, t[1], t[end])
    ylims!(ax_last, 0.0, max([maximum(mpc_data[j][:mpc_signal][3, :]) for j in 1:size(mpc_data, 1)]..., [maximum(random_data[j][:random_signal][3, :]) for j in 1:size(random_data, 1)]...) * 1.20)

    ax_arr_random = []
    for i in 1:size(random_data, 1)
        push!(ax_arr_random, Axis(fig[i, 4], aspect = 1.0, title = "$(String(colors[i]))", xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    ax_last_random = Axis(fig[1:div(size(random_data, 1), 2)  , 3], title = "Minimized Energy in Upper Right Quadrant (Random)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_last_random, t[1], t[end])
    ylims!(ax_last_random, 0.0, max([maximum(mpc_data[j][:mpc_signal][3, :]) for j in 1:size(mpc_data, 1)]..., [maximum(random_data[j][:random_signal][3, :]) for j in 1:size(random_data, 1)]...) * 1.20)
    
    mpc_vecs = [mpc_data[j][:mpc_signal][3, :] for j in 1:size(mpc_data, 1)]
    average_mpc = mean(mpc_vecs)
    upper_bound_mpc = average_mpc .+ sqrt.(var(mpc_vecs))
    lower_bound_mpc = average_mpc .- sqrt.(var(mpc_vecs))
    
    random_vecs = [random_data[j][:random_signal][3, :] for j in 1:size(random_data, 1)]
    average_random = mean(random_vecs)
    upper_bound_random = average_random .+ sqrt.(var(random_vecs))
    lower_bound_random = average_random .- sqrt.(var(random_vecs))

    ax_average = Axis(fig[(div(size(mpc_data, 1), 2)+1):size(mpc_data, 1), 2], title = "Mean and Standard Deviation (MPC)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_average, t[1], t[end])
    ylims!(ax_average, 0.0, max(maximum(upper_bound_mpc), maximum(upper_bound_random)) * 1.20)

    ax_average_random = Axis(fig[(div(size(mpc_data, 1), 2)+1):size(mpc_data, 1), 3], title = "Mean and Standard Deviation (Random)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_average_random, t[1], t[end])
    ylims!(ax_average_random, 0.0, max(maximum(upper_bound_mpc), maximum(upper_bound_random)) * 1.20)

    # Bounding box coordinates
    x1, y1 = 0, 0
    x2, y2 = 15, 15

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        for j in 1:size(mpc_data, 1)
            empty!(ax_arr[j])
            heatmap!(ax_arr[j], dim.x, dim.y, mpc_data[j][:x_mpc][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax_arr[j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(mpc_data[j][:interps_mpc]), tspan[i]))
            lines!(ax_arr[j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
        end

        for j in 1:size(random_data, 1)
            empty!(ax_arr_random[j])
            heatmap!(ax_arr_random[j], dim.x, dim.y, random_data[j][:x_random][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax_arr_random[j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(random_data[j][:interps_random]), tspan[i]))
            lines!(ax_arr_random[j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
        end

        idx = findfirst(tspan[i] .<= t)[1]
        empty!(ax_last)
        for j in 1:size(mpc_data, 1)
            lines!(ax_last, t[1:idx], mpc_data[j][:mpc_signal][3, 1:idx], color=colors[mod1(j, length(colors))])
        end

        empty!(ax_last_random)
        for j in 1:size(random_data, 1)
            lines!(ax_last_random, t[1:idx], random_data[j][:random_signal][3, 1:idx], color=colors[mod1(j, length(colors))])
        end

        empty!(ax_average)
        band!(ax_average, t[1:idx], lower_bound_mpc[1:idx], upper_bound_mpc[1:idx], color=:cyan, transparency=0.5)
        lines!(ax_average, t[1:idx], average_mpc[1:idx], color=:blue)

        empty!(ax_average_random)
        band!(ax_average_random, t[1:idx], lower_bound_random[1:idx], upper_bound_random[1:idx], color=:orange, transparency=0.5)
        lines!(ax_average_random, t[1:idx], average_random[1:idx], color=:red)
    end

end

function create_figures(mpc_data, random_data, output_path)
    colors = [  :red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown, :pink,
                :lime, :teal, :violet, :gold, :indigo, :olive, :navy, :coral, :turquoise, :salmon ]
    fig = Figure(;size = (1600, 1600))

    ax_last = Axis(fig[1:div(size(mpc_data, 1), 2)  , 1], title = "Minimized Energy in Upper Right Quadrant (MPC)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_last, t[1], t[end])
    ylims!(ax_last, 0.0, max([maximum(mpc_data[j][:mpc_signal][3, :]) for j in 1:size(mpc_data, 1)]..., [maximum(random_data[j][:random_signal][3, :]) for j in 1:size(random_data, 1)]...) * 1.20)

    ax_last_random = Axis(fig[1:div(size(random_data, 1), 2)  , 2], title = "Minimized Energy in Upper Right Quadrant (Random)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_last_random, t[1], t[end])
    ylims!(ax_last_random, 0.0, max([maximum(mpc_data[j][:mpc_signal][3, :]) for j in 1:size(mpc_data, 1)]..., [maximum(random_data[j][:random_signal][3, :]) for j in 1:size(random_data, 1)]...) * 1.20)

    mpc_vecs = [mpc_data[j][:mpc_signal][3, :] for j in 1:size(mpc_data, 1)]
    average_mpc = mean(mpc_vecs)
    upper_bound_mpc = average_mpc .+ sqrt.(var(mpc_vecs))
    lower_bound_mpc = average_mpc .- sqrt.(var(mpc_vecs))
    
    random_vecs = [random_data[j][:random_signal][3, :] for j in 1:size(random_data, 1)]
    average_random = mean(random_vecs)
    upper_bound_random = average_random .+ sqrt.(var(random_vecs))
    lower_bound_random = average_random .- sqrt.(var(random_vecs))

    ax_average = Axis(fig[(div(size(mpc_data, 1), 2)+1):size(mpc_data, 1), 1], title = "Mean and Standard Deviation (MPC)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_average, t[1], t[end])
    ylims!(ax_average, 0.0, max(maximum(upper_bound_mpc), maximum(upper_bound_random)) * 1.20)

    ax_average_random = Axis(fig[(div(size(mpc_data, 1), 2)+1):size(mpc_data, 1), 2], title = "Mean and Standard Deviation (Random)", xlabel = "Time (s)", ylabel = "Energy")
    xlims!(ax_average_random, t[1], t[end])
    ylims!(ax_average_random, 0.0, max(maximum(upper_bound_mpc), maximum(upper_bound_random)) * 1.20)

    empty!(ax_last)
    for j in 1:size(mpc_data, 1)
        lines!(ax_last, t, mpc_data[j][:mpc_signal][3, :], color=colors[mod1(j, length(colors))])
    end

    empty!(ax_last_random)
    for j in 1:size(random_data, 1)
        lines!(ax_last_random, t, random_data[j][:random_signal][3, :], color=colors[mod1(j, length(colors))])
    end

    empty!(ax_average)
    band!(ax_average, t, lower_bound_mpc, upper_bound_mpc, color=:cyan, transparency=0.5)
    lines!(ax_average, t, average_mpc, color=:blue)

    empty!(ax_average_random)
    band!(ax_average_random, t, lower_bound_random, upper_bound_random, color=:orange, transparency=0.5)
    lines!(ax_average_random, t, average_random, color=:red)

    save(output_path, fig)
end

step = 10

output_folder = mkpath("$(jobid)_$(checkpoint_step)_scattering_minimization.mpc")
prefix = "comparison"
for j in 1:step:20
    mpc_data = [BSON.load(joinpath(output_folder, "mpc_$i.bson")) for i in j:(j+step-1)]
    random_data = [BSON.load(joinpath(output_folder, "random_$i.bson")) for i in j:(j+step-1)]
    output_path = joinpath(output_folder, "$(prefix)_$j-$(j+step-1).mp4")
    # multiple_mpc_rendering(mpc_data, random_data, output_path)
    create_figures(mpc_data, random_data, joinpath(output_folder, "fig_$j-$(j+step-1).png"))
end

# step = 10
# j = 11
# mpc_data = [BSON.load(joinpath(output_folder, "mpc_$i.bson")) for i in j:(j+step-1)]
# random_data = [BSON.load(joinpath(output_folder, "random_$i.bson")) for i in j:(j+step-1)]
# p = joinpath(output_folder, "fig_$j-$(j+step-1).png")
# create_figures(mpc_data, random_data, p)