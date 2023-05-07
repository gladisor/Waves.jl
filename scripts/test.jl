include("dependencies.jl")

## Declaring some important hyperparameters
ambient_speed = AIR
## Establising the data pathway and loading in the env
data_path = "data/M=6_as=1.0_normalized"
println("Loading Env")
env = gpu(BSON.load(joinpath(data_path, "env.bson"))[:env])

println("Resetting Env")
reset!(env)
policy = RandomDesignPolicy(action_space(env))