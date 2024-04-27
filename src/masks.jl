function patch_sizes(grid_length)
    divisors = Int[]
    for i in 2:grid_length-1
        if grid_length % i == 0
            push!(divisors, i)
        end
    end
    return divisors
end

function pick_closest_number(array, chosen_number)
    closest_number = array[1]
    closest_distance = abs(chosen_number - closest_number)
    for num in array
        distance = abs(chosen_number - num)
        
        if distance < closest_distance
            closest_number = num
            closest_distance = distance
        end
    end
    return closest_number
end

function create_patches(grid_size::Int, patch_size::Int)
    patches = []
    for i in 1:patch_size:grid_size
        for j in 1:patch_size:grid_size
            mask = falses(grid_size, grid_size)
            mask[i:i+patch_size-1, j:j+patch_size-1] .= true
            push!(patches, mask)
        end
    end
    return patches
end
