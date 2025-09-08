module ControllerTypes
export MPCParams, ControllerConfig, DynamicsMode, Nominal, FullGP, LinearizedGP, Nonlinear

@enum DynamicsMode begin
    Nominal
    FullGP
    LinearizedGP
    Nonlinear
end

struct MPCParams
    horizon::Int
    control_bounds::Tuple{Float64, Float64}
    state_bounds::Tuple{Float64, Float64}
    cost_weights::Vector{Float64}
    solver_settings::Dict{String, Any}
end

struct ControllerConfig
    mpc_params::MPCParams
    use_chance_constraints::Bool
    confidence_level::Float64
    communication_radius::Float64
    safety_distance::Float64
end
end