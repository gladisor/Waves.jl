export DesignInterpolator

struct DesignInterpolator
    initial::AbstractDesign
    Δ::AbstractDesign
    ti::Float32
    tf::Float32
end

function (interp::DesignInterpolator)(t::Float32)
    d = (interp.tf - interp.ti)
    t = ifelse(d > 0.0f0, (t - interp.ti) / d, 0.0f0)
    return interp.initial + t * interp.Δ
end

function Flux.gpu(interp::DesignInterpolator)
    return DesignInterpolator(
        gpu(interp.initial),
        gpu(interp.Δ),
        interp.ti,
        interp.tf
    )
end