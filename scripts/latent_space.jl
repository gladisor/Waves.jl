using Waves

Flux.device!(1)

dataset_name = "variable_source_yaxis_x=-10.0"
DATA_PATH = "/scratch/cmpe299-fa22/tristan/data/$dataset_name"
checkpoint = 10040
our_model_name = "ours_balanced_field_scale"

OUR_MODEL_PATH = joinpath(DATA_PATH, "models/$our_model_name/checkpoint_step=$checkpoint/checkpoint.bson")
our_model = gpu(BSON.load(OUR_MODEL_PATH)[:model])

ep = Episode(path = joinpath(DATA_PATH, "episodes/episode500.bson"))
horizon = 200
s, a, t, _ = gpu(Flux.batch.(prepare_data(ep, horizon)))
s = s[1, :]
a = a[:, [1]]
t = t[:, [1]]
z = cpu(generate_latent_solution(our_model, s, a, t))

u_tot = z[:, 1, 1, :]
u_inc = z[:, 3, 1, :]
u_sc = u_tot .- u_inc

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, u_sc .^ 2, colormap = :inferno)
save("latent.png", fig)
