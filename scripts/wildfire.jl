using Waves, Flux, CairoMakie

function δ⁺(T, Tp, X_methane, X_oxygen, X_methane_e, X_oxygen_e)
    return (T .> Tp) .& (X_methane .> X_methane_e) .& (X_oxygen .> X_oxygen_e)
    # return (T .> Tp) .|| (X_methane .> X_methane_e) .|| (X_oxygen .> X_oxygen_e)
end

## order = (methane, oxygen, carbon_dioxide, nitrogen)
θ_i =   [1.0f0,     2.0f0,      1.0f0,      2.0f0,      0.0f0]
M_i =   [16.04f0,   32.00f0,    44.01f0,    18.02f0,    28.02f0]
cp_i =  [2.226f0,   0.981f0,    0.839f0,    4.1816f0,   1.040f0] * 1000.0f0
H_i =   [-74.81f0,  0.0f0,      -393.509,   -241.818,   0.0f0] * 1000.0f0

Base.@kwdef struct Wildfire <: AbstractDynamics
    grad::AbstractMatrix{Float32}

    T_amb = 298.15f0
    T_ign = 573.00f0
    Ar = 4.14f-5
    ρ = 1.2172f0
    k = 0.10f0
    Ca = 0.0600f0
    σ = 5.6704f-8
    ϵ = 0.55f0

    θ_i = permutedims(θ_i[:, :, :], (2, 3, 1))
    M_i = permutedims(M_i[:, :, :], (2, 3, 1))
    cp_i = permutedims(cp_i[:, :, :], (2, 3, 1))
    H_i = permutedims(H_i[:, :, :], (2, 3, 1))
end

Flux.@functor Wildfire

function (dyn::Wildfire)(x::AbstractArray{Float32}, t::AbstractVector{Float32}, θ)

    T = x[:, :, 1]
    X_i = x[:, :, 2:end]

    θ_i = dyn.θ_i
    M_i = dyn.M_i
    cp_i = dyn.cp_i
    H_i = dyn.H_i

    grad = dyn.grad
    T_amb, T_ign = dyn.T_amb, dyn.T_ign
    X_methane_e = 0.01f0
    X_oxygen_e = 0.01f0

    Ar = dyn.Ar
    ρ = dyn.ρ
    k = dyn.k
    Ca = dyn.Ca
    σ = dyn.σ
    ϵ = dyn.ϵ

    M = dropdims(sum(X_i .* M_i, dims = 3), dims = 3)
    cp = dropdims(sum(X_i .* M_i .* cp_i, dims = 3), dims = 3) ./ M
    hc = dropdims(sum(θ_i .* (H_i .+ M_i .* cp_i .* (T_amb .- T)[:, :, :]), dims = 3), dims = 3) ./ M

    burn = δ⁺(T, T_ign, X_i[:, :, 1], X_i[:, :, 2], X_methane_e, X_oxygen_e)
    r = -burn * Ar .* T .* sqrt.(X_i[:, :, 1]) .* X_i[:, :, 2] .* exp.(-T_ign  ./ T)
    combustion = - ρ * hc .* M ./ M_i[:, :, 1] .* r
    thermal_diffusion = k * ∂x(grad, (1.0f0 ./ cp .* ∂x(grad, cp .* T))) .+ k * ∂y(grad, (1.0f0 ./ cp .* ∂y(grad, cp .* T)))
    enthalpy_diffusion = k * ∂x(grad, (1.0f0 ./ cp .* ∂x(grad, hc))) .+ k * ∂y(grad, (1.0f0 ./ cp .* ∂y(grad, hc)))
    vertical_convection = Ca * (T_amb .- T)
    radiation = σ * ϵ * (4*∂x(grad, T.^ 3 .* ∂x(grad, T))*dx .+ 4*∂y(grad, T.^3 .* ∂y(grad, T))*dy)
    vertical_radiation = σ * ϵ * (T_amb ^ 4 .- T .^ 4)
    
    ∂T_∂t = (combustion .+ thermal_diffusion .+ enthalpy_diffusion .+ vertical_convection .+ radiation .+ vertical_radiation) ./ (ρ .* cp)

    ∂X_methane_∂t =         ( (θ_i[:, :, 1] ./ θ_i[:, :, 1]) .* M ./ M_i[:, :, 1]) .* r
    ∂X_oxygen_∂t =          ( (θ_i[:, :, 2] ./ θ_i[:, :, 1]) .* M ./ M_i[:, :, 1]) .* r
    ∂X_carbon_dioxide_∂t =  (-(θ_i[:, :, 3] ./ θ_i[:, :, 1]) .* M ./ M_i[:, :, 1]) .* r
    ∂X_water_∂t =           (-(θ_i[:, :, 4] ./ θ_i[:, :, 1]) .* M ./ M_i[:, :, 1]) .* r

    return cat(
        ∂T_∂t,
        ∂X_methane_∂t,
        ∂X_oxygen_∂t,
        ∂X_carbon_dioxide_∂t,
        ∂X_water_∂t,
        X_i[:, :, end] .* 0.0f0,
        dims = 3
    )
end

Flux.device!(0)
Flux.CUDA.allowscalar(false)

dim = TwoDim(25.0f0, 512)
dx = get_dx(dim)
dy = get_dy(dim)

dyn = gpu(Wildfire(grad = build_gradient(dim)))

ign = build_normal(build_grid(dim), [0.0f0 0.0f0; 10.0f0 -10.0f0], [1.0f0, 1.0f0], [7.0f0, 7.0f0])
T = ones(Float32, size(dim)...) * dyn.T_amb .+ ign * dyn.T_ign
X = ones(Float32, size(dim)...)
X_methane = X * 0.10f0
X_oxygen = X * 0.25f0
X_carbon_dioxide = X * 0.04f0
X_water = X * 0.01f0
X_nitrogen = X * 0.60f0

X_i = cat(X_methane, X_oxygen, X_carbon_dioxide, X_water, X_nitrogen, dims = 3)
x = gpu(cat(T, X_i, dims = 3))

# dt = 0.25f0
dt = 0.25f0
tspan = build_tspan(0.0f0, dt, 1000)

iter = gpu(Integrator(runge_kutta, dyn, dt))
@time sol = cpu(iter(x, tspan, nothing))

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0)
record(fig, "vid.mp4", axes(tspan, 1)) do i
    empty!(ax)
    heatmap!(ax, dim.x, dim.y, sol[:, :, 1, i])
end


# # # fig = Figure()
# # # ax = Axis(fig[1, 1], aspect = 1.0)
# # # hm = heatmap!(ax, dim.x, dim.y, sol[:, :, 1, 1])
# # # Colorbar(fig[1, 2], hm)
# # # ax = Axis(fig[1, 3], aspect = 1.0)
# # # hm = heatmap!(ax, dim.x, dim.y, sol[:, :, 1, end÷2])
# # # Colorbar(fig[1, 4], hm)
# # # ax = Axis(fig[1, 5], aspect = 1.0)
# # # hm = heatmap!(ax, dim.x, dim.y, sol[:, :, 1, end])
# # # Colorbar(fig[1, 6], hm)
# # # save("wildfire.png", fig)