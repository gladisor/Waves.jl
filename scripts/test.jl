using Waves, CairoMakie, Flux, BSON

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]

percentage_decrease = []
for i in 1:5
    mpc_ep1 = Episode(path = "control_results/cPILS_location=$i,episode=1.bson")
    mpc_ep2 = Episode(path = "control_results/cPILS_location=$i,episode=2.bson")
    mpc_ep3 = Episode(path = "control_results/cPILS_location=$i,episode=3.bson")
    mpc_ep4 = Episode(path = "control_results/cPILS_location=$i,episode=4.bson")

    random_ep1 = Episode(path = "control_results/random_location=$i,episode=1.bson")
    random_ep2 = Episode(path = "control_results/random_location=$i,episode=2.bson")
    random_ep3 = Episode(path = "control_results/random_location=$i,episode=3.bson")
    random_ep4 = Episode(path = "control_results/random_location=$i,episode=4.bson")

    _, _, _, mpc_y1 = prepare_data(mpc_ep1, env.actions)
    mpc_y1 = mpc_y1[1]
    _, _, _, mpc_y2 = prepare_data(mpc_ep2, env.actions)
    mpc_y2 = mpc_y2[1]
    _, _, _, mpc_y3 = prepare_data(mpc_ep3, env.actions)
    mpc_y3 = mpc_y3[1]
    _, _, _, mpc_y4 = prepare_data(mpc_ep4, env.actions)
    mpc_y4 = mpc_y4[1]

    _, _, _, random_y1 = prepare_data(random_ep1, env.actions)
    random_y1 = random_y1[1]
    _, _, _, random_y2 = prepare_data(random_ep2, env.actions)
    random_y2 = random_y2[1]
    _, _, _, random_y3 = prepare_data(random_ep3, env.actions)
    random_y3 = random_y3[1]
    _, _, _, random_y4 = prepare_data(random_ep4, env.actions)
    random_y4 = random_y4[1]

    mpc_y = (mpc_y1 .+ mpc_y2 .+ mpc_y3 .+ mpc_y4) / 4
    random_y = (random_y1 .+ random_y2 .+ random_y3 .+ random_y4) / 4

    mpc_y_avg = Flux.mean(mpc_y[end÷2:end, 3])
    random_y_avg = Flux.mean(random_y[end÷2:end, 3])

    push!(
        percentage_decrease, 
        (random_y_avg - mpc_y_avg) / random_y_avg)
end

# bson("cPILS_percentage_decrease.bson"; :percentage_decrease => percentage_decrease)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, random_y[:, 3], label = "Random")
# lines!(ax, mpc_y[:, 3], label = "MPC")
# axislegend(ax)
# save("control_loc=$i.png", fig)


