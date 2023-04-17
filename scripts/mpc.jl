
using Waves
# ### MPC COST
# # s = gpu(state(env))
# # control_sequence = gpu(build_control_sequence(initial_design, 2))

# # opt_state = Optimisers.setup(Optimisers.Momentum(1e-5), control_sequence)

# # for i in 1:10
# #     cost, back = pullback(a -> build_mpc_cost(model, s, a), control_sequence)
# #     gs = back(one(cost))[1]
# #     opt_state, control_sequence = Optimisers.update(opt_state, control_sequence, gs)
# #     println(cost)
# # end