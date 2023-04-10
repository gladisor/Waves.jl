export plot_solution!

function plot_solution!(nrows::Int, ncols::Int, dim::TwoDim, u::AbstractArray{Float32, 4}; path::String, field::Int = 1)
    
    fig = Figure()
    layout = fig[1, 1] = GridLayout(nrows, ncols)

    steps = size(u, ndims(u))
    n = nrows * ncols
    idx = Int.(round.(LinRange(1, steps, n)))

    for i in 1:nrows
        for j in 1:ncols
            k = (i-1) * ncols + j
            ax = Axis(layout[i, j], aspect = 1.0f0)
            heatmap!(ax, dim.x, dim.y, u[:, :, field, idx[k]], colormap = :ice)
        end
    end

    save(path, fig)
end