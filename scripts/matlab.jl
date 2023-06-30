using MAT
using BSON
using ReinforcementLearning
using Waves

main_path = "data/triple_ring_dataset"
data_path = joinpath(main_path, "episodes")
println("Loading Environment")
env = BSON.load(joinpath(data_path, "env.bson"))[:env]
s = state(env)
file = matopen("triple_ring.mat", "w")
xM_config = hcat(s.design.config.cylinders.pos, s.design.config.cylinders.r)
xM_core = hcat(s.design.core.pos, s.design.core.r)
xM = vcat(xM_core, xM_config)
write(file, "xM", xM)
close(file)


main_path = "/scratch/cmpe299-fa22/tristan/data/single_cylinder_dataset"
data_path = joinpath(main_path, "episodes")
println("Loading Environment")
env = BSON.load(joinpath(data_path, "env.bson"))[:env]
s = state(env)
file = matopen("single_scatterer.mat", "w")
xM_config = hcat(s.design.config.cylinders.pos, s.design.config.cylinders.r)
xM_core = hcat(s.design.core.pos, s.design.core.r)
xM = vcat(xM_core, xM_config)
write(file, "xM", xM)
close(file)