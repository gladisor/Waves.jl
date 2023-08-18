using CairoMakie
using BSON

Flux.device!(1)

main_path = "/scratch/cmpe299-fa22/tristan/data/actions=200_design_space=build_triple_ring_design_space_freq=1000.0"
pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=10000.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
no_pml_model_path = joinpath(main_path, "models/SinWaveEmbedderV11/horizon=20_nfreq=200_pml=0.0_lr=0.0001_batchsize=32/epoch_90/model.bson")
data_path = joinpath(main_path, "episodes")

batchsize = 32
horizons = collect(20:10:200)

println("Loading Environment")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])
dim = cpu(env.dim)
reset!(env)
policy = RandomDesignPolicy(action_space(env))

println("Preparing Data")
train_data = Vector{EpisodeData}([EpisodeData(path = joinpath(data_path, "episode$i/episode.bson")) for i in 961:1000])

println("Loading Models")
pml_model = gpu(BSON.load(pml_model_path)[:model])
no_pml_model = gpu(BSON.load(no_pml_model_path)[:model])
testmode!(pml_model)
testmode!(no_pml_model)

loss = Dict(
    :pml => Vector{Float32}[],
    :no_pml => Vector{Float32}[]
)

for h in horizons
    train_loader = DataLoader(prepare_data(train_data, h), shuffle = true, batchsize = batchsize, partial = false)
    states, actions, tspans, sigmas = gpu(first(train_loader))

    println("Evaluating on Batch")
    @time begin
        pml_sigmas = pml_model(states, actions, tspans)
        no_pml_sigmas = no_pml_model(states, actions, tspans)
    end

    y = flatten_repeated_last_dim(sigmas)
    pml_loss = cpu(vec(Flux.mse(pml_sigmas, y, agg = x -> Flux.mean(x, dims = 1))))
    no_pml_loss = cpu(vec(Flux.mse(no_pml_sigmas, y, agg = x -> Flux.mean(x, dims = 1))))

    push!(loss[:pml], pml_loss)
    push!(loss[:no_pml], no_pml_loss)
end

pml_loss = hcat(loss[:pml]...)
no_pml_loss = hcat(loss[:no_pml]...)

BSON.bson(
    "loss.bson", 
    horizon = horizons,
    pml_loss = pml_loss, 
    no_pml_loss = no_pml_loss)

data = BSON.load("loss.bson")
horizon = data[:horizon]
pml_loss = data[:pml_loss]
no_pml_loss = data[:no_pml_loss]

xs = vec(ones(Int, size(pml_loss)) .* horizon')

dodge = ones(Int, size(pml_loss))
dodge1 = vec(dodge)
dodge2 = 2 * vec(dodge)

fig = Figure()
ax = Axis(fig[1, 1], 
    title = "Effect of Increased Planning Horizon on Validation Loss",
    xlabel = "Planning Horizon", 
    ylabel = "Validation Loss")

boxplot!(
    ax, 
    xs, 
    vec(pml_loss),
    dodge = dodge1, 
    show_outliers = false,
    color = :red,
    width = 3,
    label = "PML")

boxplot!(
    ax,
    xs, 
    vec(no_pml_loss),
    dodge = dodge2,
    show_outliers = false,
    color = :blue,
    width = 3,
    label = "No PML")

axislegend(ax, position = :lt)
save("loss.png", fig)

