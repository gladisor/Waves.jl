struct DesignEncoder
    dense1::Dense
    dense2::Dense
    dense3::Dense
    dense4::Dense
end

Flux.@functor DesignEncoder

function DesignEncoder(in_size::Int, h_size::Int, out_size::Int, activation::Function)
    dense1 = Dense(in_size, h_size, activation)
    dense2 = Dense(h_size, h_size, activation)
    dense3 = Dense(h_size, h_size, activation)
    dense4 = Dense(h_size, out_size, sigmoid)
    return DesignEncoder(dense1, dense2, dense3, dense4)
end

function (encoder::DesignEncoder)(design::AbstractDesign, action::AbstractDesign)
    x = vcat(vec(design), vec(action))
    return x |> encoder.dense1 |> encoder.dense2 |> encoder.dense3 |> encoder.dense4
end