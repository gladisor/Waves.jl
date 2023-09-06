export WaveEnvState
export WaveImage, DisplacementImage
export WaveEnv, RandomDesignPolicy

# function Base.convert(
#         ::Type{CircularBuffer{AbstractArray{Float32, 3}}}, 
#         x::Union{
#             Vector{CUDA.CuArray{Float32, 3, CUDA.Mem.DeviceBuffer}},
#             Vector{Array{Float32, 3}}
#             }
#         )

#     cb = CircularBuffer{AbstractArray{Float32, 3}}(length(x))
#     [push!(cb, x[i]) for i in axes(x, 1)]
#     return cb
# end

struct WaveEnvState
    dim::TwoDim
    tspan::Vector{Float32}

    wave::AbstractArray{Float32, 3}
    design::AbstractDesign
end

Flux.@functor WaveEnvState

mutable struct WaveEnv <: AbstractEnv
    dim::AbstractDim
    
    design_space::DesignSpace
    design::AbstractDesign ## state of design
    wave::AbstractArray{Float32} ## state of wave
    source::AbstractSource ## source of energy in environment

    iter::Integrator

    signal::AbstractArray{Float32} ## information saved from integration
    time_step::Int ## time in environment

    ## static parameters
    resolution::Tuple{Int, Int}
    action_speed::Float32 ## speed that action is applied over m/s
    dt::Float32
    integration_steps::Int
    actions::Int
end

Flux.@functor WaveEnv
Flux.trainable(::WaveEnv) = (;)

function WaveEnv(
        dim::TwoDim;
        design_space::DesignSpace,
        action_speed::Float32 = 500.0f0,
        source::AbstractSource = NoSource(),

        c0::Float32 = WATER,
        pml_width::Float32 = 2.0f0,
        pml_scale::Float32 = 20000.0f0,

        resolution::Tuple{Int, Int} = (128, 128), ## this must be less than 
        dt::Float32 = 1f-5,
        integration_steps::Int = 100,
        actions::Int = 10)

    @assert all(size(dim) .> resolution) "Resolution must be less than finite element grid."

    wave = build_wave(dim, 6)
    design = rand(design_space)
    dyn = AcousticDynamics(dim, c0, pml_width, pml_scale)
    iter = Integrator(runge_kutta, dyn, dt, integration_steps)
    signal = zeros(Float32, integration_steps + 1)

    return WaveEnv(
        dim, 
        design_space, design, 
        wave, source, 
        iter, signal, 0, 
        resolution, action_speed, dt, integration_steps, actions)
end

function Base.time(env::WaveEnv)
    return env.time_step * env.dt
end

function Waves.build_tspan(env::WaveEnv) 
    return build_tspan(time(env), env.dt, env.integration_steps)
end

function RLBase.is_terminated(env::WaveEnv)
    return env.time_step >= env.actions * env.integration_steps
end

function RLBase.reset!(env::WaveEnv)
    env.time_step = 0

    z = gpu(zeros(Float32, size(env.wave_total[end])))

    empty!(env.wave_total)
    empty!(env.wave_incident)

    fill!(env.wave_total, z)
    fill!(env.wave_incident, z)

    push!(env.wave_total, env.reset_wave(z))
    push!(env.wave_incident, env.reset_wave(z))

    env.total_dynamics = update_design(
        env.total_dynamics, 
        DesignInterpolator(rand(env.design_space)) ## randomly sample design from ds
        )

    env.signal *= 0.0f0
    return nothing
end

function (env::WaveEnv)(action::AbstractDesign)
    tspan = build_tspan(env)
    ti = time(env)

    current_design = env.design
    next_design = env.design_space(current_design, action)
    interp = DesignInterpolator(current_design, next_design, ti, tspan[end])

    grid = build_grid(env.dim)
    C = t -> speed(interp(cpu(t)[1]), grid, env.iter.dynamics.c0)

    sol = env.iter(env.wave, gpu(tspan), [C, env.source])
    u_tot = sol[:, :, 1, :]

    env.design = next_design
    env.wave = sol[:, :, :, end]
    env.time_step += env.integration_steps
    return tspan, interp, u_tot
    
    # u_scattered = u_total .- u_incident
    # env.signal = sum.(energy.(displacement.(u_scattered))) * get_dx(env.dim) * get_dy(env.dim)
    # return (tspan, cpu(u_incident), cpu(u_scattered))
end

function RLBase.state(env::WaveEnv)

    env.wave

    # d = imresize(
    #     cpu(x[:, :, 1, :]), ## extracting displacement field of 2d membrane
    #     env.resolution
    #     )

    # return WaveEnvState(
    #     cpu(env.dim),
    #     build_tspan(env), ## forward looking tspan
    #     d,
    #     cpu(env.total_dynamics.design(time(env))) ## getting the current design
    #     )
end

function RLBase.state_space(env::WaveEnv)
    return state(env)
end

function RLBase.action_space(env::WaveEnv)
    return build_action_space(rand(env.design_space), env.action_speed * env.dt * env.integration_steps)
end

function RLBase.reward(env::WaveEnv)
    return sum(env.signal)
end

mutable struct RandomDesignPolicy <: AbstractPolicy
    a_space::DesignSpace
end

function (policy::RandomDesignPolicy)(::WaveEnv)
    return rand(policy.a_space)
end