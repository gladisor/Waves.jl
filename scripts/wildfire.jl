using Waves, Flux, CairoMakie

function δ⁺(T, Tp, X_methane, X_oxygen, X_methane_e, X_oxygen_e)
    return (T .> Tp) .& (X_methane .> X_methane_e) .& (X_oxygen .> X_oxygen_e)
end

struct Wildfire <: AbstractDynamics

end

Flux.@functor Wildfire

function (dyn::Wildfire)(x::AbstractArray, t::AbstractVector{Float32}, θ)

    # T = x[:, :, 1]
    # X_i = x[:, :, 2:end]

    # M = dropdims(sum(X_i .* M_i, dims = 3), dims = 3)
    # cp = dropdims(sum((X_i .* M_i .* cp_i) ./ M[:, :, :], dims = 3), dims = 3)
    # hc = dropdims(sum(θ_i .* (H_i .+ M_i .* cp_i .* (T_amb .- T)[:, :, :]), dims = 3), dims = 3) ./ M

    # burn = δ⁺(T, T_ign, X_methane, X_oxygen, X_methane_e, X_oxygen_e)
    # r = -burn * Ar .* T .* sqrt.(X_methane) .* X_oxygen .* exp.(-T_amb ./ T)
    # combustion = - ρ * hc .* M ./ M_methane .* r
    # thermal_diffusion = k * ∂x(grad, (1 ./ cp .* ∂x(grad, cp .* T))) .+ k * ∂y(grad, (1 ./ cp .* ∂y(grad, cp .* T)))
    # enthalpy_diffusion = k * ∂x(grad, (1 ./ cp .* ∂x(grad, hc))) .+ k * ∂y(grad, (1 ./ cp .* ∂y(grad, hc)))
    # vertical_convection = Ca * (T_amb .- T)
    # radiation = σ * ϵ * (4*∂x(grad, T.^3 .* ∂x(grad, T))*dx .+ 4*∂y(grad, T.^3 .* ∂y(grad, T))*dy)
    # vertical_radiation = σ * ϵ * (T_amb ^ 4 .- T .^ 4)

    # ∂T_∂t = combustion .+ thermal_diffusion .+ enthalpy_diffusion .+ vertical_convection .+ radiation .+ vertical_radiation
    # ∂X_methane_∂t = ((θ_methane / θ_methane) .* M / M_methane) .* r
    # ∂X_oxygen_∂t = ((θ_oxygen / θ_methane) .* M / M_methane) .* r
    # ∂X_carbon_dioxide_∂t = (-(θ_carbon_dioxide / θ_methane) .* M / M_methane) .* r
    # ∂X_water_∂t = (-(θ_water / θ_methane) .* M / M_methane) .* r

    # return cat(
    #     ∂T_∂t,
    #     ∂X_methane_∂t,
    #     ∂X_oxygen_∂t,
    #     ∂X_carbon_dioxide_∂t,
    #     ∂X_water_∂t,
    #     X_i[:, :, end] .* 0.0f0,
    #     dims = 3
    # )

    return (∂x(grad, x) .+ ∂y(grad, x))
end

# Flux.device!(0)
# Flux.CUDA.allowscalar(false)

dim = TwoDim(100.0f0, 700)
dx = get_dx(dim)
dy = get_dy(dim)
grad = build_gradient(dim)

T_amb = 298.15f0
T_ign = 573.00f0
Ar = 4.14f-5
ρ = 1.2172f0
k = 0.10f0
Ca = 0.0600
σ = 5.6704f-8
ϵ = 0.55f0

ign = build_normal(build_grid(dim), [0.0f0   0.0f0; 10.0f0 -10.0f0], [1.0f0, 1.0f0], [7.0f0, 7.0f0])
# T = ones(Float32, size(dim)...) * T_amb .+ ign * T_ign
T = ign * T_ign

## order = (methane, oxygen, carbon_dioxide, nitrogen)
θ_i = [1, 2, 1, 2, 0]
θ_i = permutedims(θ_i[:, :, :], (2, 3, 1))

M_methane = 16.04f0
M_oxygen = 32.00f0
M_carbon_dioxide = 44.01f0
M_water = 18.02f0
M_nitrogen = 28.02f0
M_i = [16.04f0, 32.00f0, 44.01f0, 18.02f0, 28.02f0]
M_i = permutedims(M_i[:, :, :], (2, 3, 1))

X = ones(Float32, size(dim)...)
X_methane = X * 0.10f0
X_methane_e = 0.01f0
X_oxygen = X * 0.25f0
X_oxygen_e = 0.01
X_carbon_dioxide = X * 0.04f0
X_water = X * 0.01f0
X_nitrogen = X * 0.60f0
X_i = cat(X_methane, X_oxygen, X_carbon_dioxide, X_water, X_nitrogen, dims = 3)

cp_i = [2.226f0, 0.981f0, 0.839f0, 4.1816f0, 1.040f0]
cp_i = permutedims(cp_i[:, :, :], (2, 3, 1))

H_i = [-74.81f0, 0.0f0, -393.509, -241.818, 0.0f0]
H_i = permutedims(H_i[:, :, :], (2, 3, 1))

dyn = Wildfire()
# x = cat(T, X_i, dims = 3)
x = T

dt = 0.02f0
tspan = build_tspan(0.0f0, dt, 10)
iter = Integrator(runge_kutta, dyn, dt)
# dyn(x, tspan[1, :], nothing)
sol = iter(x, tspan, nothing)



fig = Figure()
ax = Axis(fig[1, 1], aspect = 1.0)
hm = heatmap!(ax, dim.x, dim.y, sol[:, :, end])
Colorbar(fig[1, 2], hm)
save("wildfire.png", fig)



# fig = Figure()

# ax = Axis(fig[1, 1], aspect = 1.0)
# hm = heatmap!(ax, dim.x, dim.y, sol[:, :, 1, 1])
# Colorbar(fig[1, 2], hm)

# ax = Axis(fig[1, 3], aspect = 1.0)
# hm = heatmap!(ax, dim.x, dim.y, sol[:, :, 1, end÷2])
# Colorbar(fig[1, 4], hm)

# ax = Axis(fig[1, 5], aspect = 1.0)
# hm = heatmap!(ax, dim.x, dim.y, sol[:, :, 1, end])
# Colorbar(fig[1, 6], hm)

# save("wildfire.png", fig)