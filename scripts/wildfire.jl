using Waves, Flux, CairoMakie

function δ⁺(T::AbstractArray{Float32}, T_ign::Float32, X₁₂::AbstractArray{Float32}, X₁₂ₑ::AbstractArray{Float32})
    has_fuel_or_oxidizer = (X₁₂ .> X₁₂ₑ)
    return (T .> T_ign) .| (has_fuel_or_oxidizer[:, :, 1] .& has_fuel_or_oxidizer[:, :, 2])
end

function unsqueeze_vec_2d(x::AbstractVector)
    return permutedims(x[:, :, :], (2, 3, 1))
end

function heat_capacity(coefs::AbstractVector{Float32}, T::AbstractMatrix{Float32})
    T̃ = cat([T .^ i for i in 0:(length(coefs)-1)]..., dims = ndims(T) + 1)
    return dropdims(sum(permutedims(coefs[:, :, :], (2, 3, 1)) .* T̃, dims = 3), dims =  3)
end

function heat_capacity(coefs::AbstractVector{Float32}, T::Float32)
    T̃ = cat([T .^ i for i in 0:(length(coefs)-1)]..., dims = ndims(T) + 1)
    return coefs' * T̃
end

function average_heat_capacity(coefs::AbstractVector{Float32}, t_min::Float32, t_max::Float32, nt::Int)
    t = collect(range(t_min, t_max, nt))
    heat_capacity.([coefs], t)
end

Flux.device!(0)
Flux.CUDA.allowscalar(false)

## defining parameters
ρ = 1.2172f0
cₕ = 0.803f0
Aᵣ = 4.154f-5
k = 0.255f0
δ_x = 2.89f-2
δ_z = 1.91f0
χ = 2.16f-2
T_amb = 298.15f0
T_ref = T_amb
T_ign = 431.60f0
T_a = T_ign
R = 8.3144598f0 ## universal gas constant

CH4_a = [5.14987613f0, -1.36709788f-2, 4.9100599f-5, -4.84743026f-8, 1.66693956f-11]
O2_a  = [3.78246636f0, -2.99673415f-3, 9.84730200f-6, -9.68129608f-9, 3.24372836f-12]
CO2_a = [2.35677352f0, 8.98459677f-3, -7.12356269f-6, 2.45919022f-9, -1.43699548f-13]
H2O_a = [4.19864056f0, -2.03643410f-3, 6.52040211f-6, -5.48797062f-9, 1.77197817f-12]
N2_a  = [3.53100528f0, -1.23660987f-4, -5.02999437f-7, 2.43530612f-9, -1.40881235f-12]

t = collect(range(T_amb, 1100.0f0, 1000))

cp_CH4 = Flux.mean(heat_capacity.([CH4_a], t))
cp_O2 = Flux.mean(heat_capacity.([O2_a], t))
cp_CO2 = Flux.mean(heat_capacity.([CO2_a], t))
cp_H2O = Flux.mean(heat_capacity.([H2O_a], t))
cp_N2 = Flux.mean(heat_capacity.([N2_a], t))
cp_i = R * unsqueeze_vec_2d([cp_CH4, cp_O2, cp_CO2, cp_H2O, cp_N2])



# θ_i =   unsqueeze_vec_2d([1.0f0,     2.0f0,      1.0f0,      2.0f0,      0.0f0])
# M_i =   unsqueeze_vec_2d([16.04f0,   32.00f0,    44.01f0,    18.02f0,    28.02f0])
# mf_0 = unsqueeze_vec_2d([0.1f0, 0.2016f0, 0.0129f0, 0.0108f0, 0.6747f0])
# X₁₂ₑ = unsqueeze_vec_2d([0.01f0, 0.01f0])

# ## defining space
# dim = TwoDim(100.0f0, 512)
# dx = get_dx(dim)
# dy = get_dy(dim)
# grid = build_grid(dim)
# fire_centers = [0.0f0   0.0f0; 10.0f0 -10.0f0]
# num_centers = size(fire_centers, 1)

# ## initial conditions
# T = build_normal(grid, fire_centers, fill(1.0f0, num_centers), fill(1.0f0, num_centers))
# T = T_amb .+ T .* (T_ign .+ 300.0f0) ./ maximum(T)
# X_i = ones(Float32, size(dim)..., length(mf_0)) .* mf_0

# M̄ = dropdims(sum(X_i .* M_i, dims = 3), dims = 3)
# cp = dropdims(sum((X_i .* M_i .* cp_i) ./ M̄, dims = 3), dims = 3)
# r = -δ⁺(T, T_ign, X_i[:, :, 1:2], X₁₂ₑ) * Aᵣ .* T .* sqrt.(X_i[:, :, 1]) .* X_i[:, :, 2] .* exp.(- T_a ./ T)




# # fig = Figure()
# # ax = Axis(fig[1, 1], aspect = 1.0f0)
# # hm = heatmap!(ax, dim.x, dim.y, r)
# # Colorbar(fig[1, 2], hm)
# # save("wildfire.png", fig)