
# # function build_mask(dim::TwoDim)

# # end

# # g = grid(dim)
# # mask = map(xy -> xy[1]^2 + xy[2]^2 <= 9, g)

# # flux = Float64[]
# # for t ∈ range(tspan..., 200)
# #     push!(
# #         flux, 
# #         sum((∇x(∇x(sol(t)[:, :, 1], Δ), Δ) .+ ∇y(∇y(sol(t)[:, :, 1], Δ), Δ)) .* mask)
# #         )
# # end

# # save(plot(collect(range(tspan..., 200)), flux), "flux.png")