export DesignInterpolator

struct DesignInterpolator
    initial::AbstractDesign
    Δ::AbstractDesign
    ti::Float32
    tf::Float32
end

function DesignInterpolator(initial::AbstractDesign)
    return DesignInterpolator(initial, zero(initial), 0.0f0, 0.0f0)
end

function DesignInterpolator(initial::Nothing)
    return initial
end

function (interp::DesignInterpolator)(t::Float32)
    d = (interp.tf - interp.ti)
    t = ifelse(d > 0.0f0, (t - interp.ti) / d, 0.0f0)
    return interp.initial + t * interp.Δ
end

function Flux.gpu(design::DesignInterpolator)
    return DesignInterpolator(
        gpu(design.initial),
        gpu(design.Δ),
        design.ti,
        design.tf
    )
end

function Flux.cpu(design::DesignInterpolator)
    return DesignInterpolator(
        cpu(design.initial),
        cpu(design.Δ),
        design.ti,
        design.tf
    )
end

function initial_design(design::DesignInterpolator)
    return design(design.ti)
end

function final_design(design::DesignInterpolator)
    return design(design.tf)
end