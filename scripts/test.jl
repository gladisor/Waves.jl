using Waves
using Waves: get_domain, spacetime, signature, wave_discretizer, wave_equation, unpack, WaveBoundary, perturb
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, IfElse

gs = 5.0
dim = TwoDim(size = gs)
wave = Wave(dim = dim)

kwargs = Dict(
    :wave => wave, 
    :ic => GaussianPulse(1.0, loc = [2.5, 2.5]), 
    :boundary => MinimalBoundary(), 
    :ambient_speed => 1.0, 
    :tmax => 20.0, :n => 21, :dt => 0.05)

M = 3

# println("Making Design")
# @time design = Waves.ParameterizedDesign(Configuration(dim, M = M, r = 0.5, c = 0.0))

# println("Building Simulator")
# @time sim = WaveSim(design = design; kwargs...)

# println("Making Env")
# @time env = WaveEnv(sim = sim, design = design, design_steps = 20)
# design_trajectory = Vector{typeof(env.design.design)}([env.design.design])

# println("Running Env Loop")
# @time while !is_terminated(env)
#     action = Configuration(dim, M = M, r = 0.0, c = 0.0) / 5
#     steps = perturb(env, action)
#     [push!(design_trajectory, s) for s in steps]
# end

# println("Rendering")
# @time render!(WaveSol(env), design = design_trajectory, path = "test.mp4")

function (design::ParameterizedDesign)(t::Real)
    return interpolate(design.initial, design.final, t)
end

function Waves.wave_speed(wave::Wave{TwoDim}, design::Waves.ParameterizedDesign{Configuration})::Function
    C = (x, y, t) -> begin
        design = Waves.interpolate(design.initial, design.final, Waves.get_t_norm(design, t))
        in_design = (x, y) ∈ design
        return in_design' * speeds(design) + (1 - sum(in_design)) * wave.speed
    end

    return C
end

function (design::ParameterizedDesign)(t::Real)
    return interpolate(design.initial, design.final, t)
end

x, y, t, u = unpack(wave)
Dx = Differential(x); Dxx = Differential(x)^2
Dy = Differential(y); Dyy = Differential(y)^2
Dt = Differential(t); Dtt = Differential(t)^2
# C = eval(build_function(Waves.wave_speed(wave, design), Waves.spacetime(wave), parallel = MultithreadedForm()))
C = Waves.wave_speed(wave, design)


eq = Dtt(u(x, y, t)) ~ C(x, y, t)^2 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t)))
# eq = wave_equation(wave)
ps = [wave.speed => kwargs[:ambient_speed]]
bcs = [
    wave.u(dims(wave)..., 0.0) ~ kwargs[:ic](wave),
    kwargs[:boundary](wave)...
    ]

@named sys = PDESystem(eq, bcs, get_domain(wave, tmax = kwargs[:tmax]), spacetime(wave), [signature(wave)], ps)
disc = wave_discretizer(wave, kwargs[:n])
iter = init(discretize(sys, disc), Midpoint(), advance_to_tstop = true, saveat = kwargs[:dt])
sim = WaveSim(wave, get_discrete(sys, disc), iter, kwargs[:dt])
propagate!(sim)
render!(WaveSol(sim), path = "test.mp4")
