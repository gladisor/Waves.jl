export WaveEnvState
export WaveImage, DisplacementImage
export WaveEnv, RandomDesignPolicy

struct WaveEnvState
    dim::TwoDim
    tspan::Vector{Float32}
    wave::AbstractArray{Float32}
    design::AbstractDesign
end

Flux.@functor WaveEnvState

mutable struct WaveEnv <: AbstractEnv
    dim::TwoDim
    
    design_space::DesignSpace
    design::AbstractDesign          ## state of design
    wave::AbstractArray{Float32}    ## state of wave
    source::AbstractSource          ## source of energy in environment
    iter::Integrator

    signal::AbstractArray{Float32}  ## information saved from integration
    time_step::Int                  ## time in environment

    ## static parameters
    resolution::Tuple{Int, Int}
    action_speed::Float32           ## speed that action is applied over m/s
    dt::Float32
    integration_steps::Int
    actions::Int
end

Flux.@functor WaveEnv
Flux.trainable(::WaveEnv) = (;)

function WaveEnv(
        dim::TwoDim;
        design_space::DesignSpace,
        action_speed::Float32 = 250.0f0,
        source::AbstractSource = NoSource(),

        c0::Float32 = WATER,
        pml_width::Float32 = 2.0f0,
        pml_scale::Float32 = 20000.0f0,

        resolution::Tuple{Int, Int} = (128, 128), ## this must be less than 
        dt::Float32 = 1f-5,
        integration_steps::Int = 100,
        actions::Int = 10)

    @assert all(size(dim) .> resolution) "Resolution must be less than finite element grid."

    wave = zeros(Float32, size(dim)..., 12, 3)       ## initialize wave state
    design = rand(design_space)                     ## initialize design state
    signal = zeros(Float32, integration_steps + 1)  ## initialize signal quanitity

    dyn = AcousticDynamics(dim, c0, pml_width, pml_scale)
    iter = Integrator(runge_kutta, dyn, dt)

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
    env.wave *= 0.0f0
    env.design = rand(env.design_space)
    env.signal *= 0.0f0
    return nothing
end

FRAMESKIP = 10
function (env::WaveEnv)(action::AbstractDesign)
    tspan = build_tspan(env)
    ti = time(env)

    current_design = env.design
    next_design = env.design_space(current_design, action)
    interp = DesignInterpolator(current_design, next_design, ti, tspan[end])
    grid = build_grid(env.dim) ## need to fix this
    C = t -> speed(interp(cpu(t)[1]), grid, env.iter.dynamics.c0)

    ## integrating dynamics
    sol = env.iter(env.wave[:, :, :, end], gpu(tspan), [C, env.source])

    ## seperation of scattered energy
    u_tot = sol[:, :, 1, :]
    u_inc = sol[:, :, 7, :]
    u_sc = u_tot .- u_inc
    dΩ = get_dx(env.dim) * get_dy(env.dim)
    tot_energy = vec(sum(u_tot .^ 2, dims = (1, 2))) * dΩ
    inc_energy = vec(sum(u_inc .^ 2, dims = (1, 2))) * dΩ
    sc_energy =  vec(sum(u_sc  .^ 2, dims = (1, 2))) * dΩ

    ## setting environment variables
    env.signal = hcat(tot_energy, inc_energy, sc_energy)
    env.design = next_design
    env.wave = sol[:, :, :, end-(2*FRAMESKIP):FRAMESKIP:end] ## 3 frames with frameskip of 5
    env.time_step += env.integration_steps

    ## returning some internal information which is not stored
    return tspan, cpu(interp), cpu(u_tot), cpu(u_inc)
end

function RLBase.state(env::WaveEnv)
    ## only the total wave is observable
    u_tot = imresize(cpu(env.wave[:, :, 1, :]), env.resolution)
    return WaveEnvState(cpu(env.dim), build_tspan(env), u_tot, cpu(env.design))
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